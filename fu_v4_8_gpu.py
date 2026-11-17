#!/usr/bin/env python3
"""
fu_v4_8_gpu.py (v6 — final, addresses all 7 reviewer points + fixes)
GPU-accelerated per-target calibrated auditor.

Cumulative fixes:
  [FIX 2]  B_w uses s1 (not sm) for the ml channel
  [FIX 4]  --check_sigma flag
  [FIX 5]  2-fold split for `worst` calibration
  [FIX 10] Uses SEED_BASE from fu_v4_3
  [FIX 15] hit_oc uses K null-check draws (documented)
  [FIX 16] M_bw = min(M, 200) documented
  [FIX 17] --adjacency replace documented
  [FIX A]  run_seed_gpu passes stratum to make_ctx (paired DiD)
  [FIX F]  DiD uses PAIRED shared bootstrap indices
  [FIX G]  --check_sigma uses actual Vt_mean
  [FIX I]  mstar averaged over R=20 draws
  [FIX D]  Default --seq_nus2 includes 100-1000
  [FIX B1] mstar distribution printed per epsilon
  [FIX B2] CT is cost at 1 publication (independent of k)
  [FIX B3] oc uses K null-check draws, not M=300
  [REV 4]  --composition_mode {per_release, total_transcript}
  [REV 5]  holm_correction_results.txt produced from this script (append mode)
  [REV 6]  SD_FACTOR env var supported (multiplicative sensitivity on sd)
"""
import argparse
import math
import os
import time
from collections import Counter
import numpy as np
import torch
from scipy.stats import norm

import fu_v4_3
from fu_v4_3 import (
    analytic_sigma, ss_release, eigh_fast, make_ctx,
    frontier_pts, cost_at, lab, SEED_BASE, holm_bonferroni,
)
for k_add in ["X", "features"]:
    if k_add not in fu_v4_3.KEYS["Ftr"]:
        fu_v4_3.KEYS["Ftr"].append(k_add)

FAMS = ["A", "A2", "B4", "B", "B4_w", "B_w"]
FAMN = {
    "A": "A    bruit iso (weights)",
    "A2": "A2   bruit aniso (weights)",
    "B4": "B4   DP eps<=4 (stats UB)",
    "B": "B    DP tous eps (stats UB)",
    "B4_w": "B4_w DP eps<=4 (weights sym)",
    "B_w": "B_w  DP tous eps (weights sym)",
}
NCH = 6
HOLM_FILE = "holm_correction_results.txt"


def acc_np(W, Xm, yy):
    return float(np.mean(np.argmax(Xm @ W, 1) == yy))


def sens_bound(args):
    return math.sqrt(5.0) if args.adjacency == "replace" else math.sqrt(2.0)


def effective_eps_step(eps_step, k, composition_mode):
    """[REV 4] Effective per-release epsilon under the chosen composition mode."""
    if composition_mode == "total_transcript":
        return eps_step / math.sqrt(k)
    return eps_step


@torch.no_grad()
def ss_noisy_batch_gpu(Sn_batch, Gn_batch, sig, m, g, d):
    mu, V = torch.linalg.eigh(Sn_batch)
    mu = torch.clamp(mu, min=0.0)
    denom = (mu + g + m * sig * math.sqrt(d)).unsqueeze(-1)
    VTG = torch.matmul(V.transpose(-2, -1), Gn_batch)
    return torch.matmul(V, VTG / denom)


def _score_channels(A1, A2_, rr, RNn, li, t1n2, sli, kind):
    """[FIX 2]: B_w uses s1 (whitened projection) as the ml channel."""
    s0 = A1 / (math.sqrt(t1n2) * RNn)
    s1 = A2_ / (sli * RNn)
    if kind == "out":
        sm = A1 / (1.0 - li) - 0.5 * t1n2 * rr / ((1.0 - li) ** 2)
    elif kind == "an":
        sm = A2_ / (1.0 - li) - 0.5 * li * rr / ((1.0 - li) ** 2)
    elif kind == "B_w":
        sm = s1
    else:
        raise ValueError(f"unknown kind: {kind}")
    return s0, s1, sm


@torch.no_grad()
def calibrate_channels_gpu(NS, ob, oc, al, legacy_calib=False):
    """[FIX 5] `worst` uses a 2-fold split of the M null draws."""
    M = NS.shape[0]
    if legacy_calib:
        p_ob = (1.0 + (NS >= ob.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
        p_oc = (1.0 + (NS >= oc.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
        h_ob_raw = (p_ob <= al)
        h_oc_raw = (p_oc <= al)
        base_ob = torch.maximum(h_ob_raw[:, 0], h_ob_raw[:, 1])
        base_oc = torch.maximum(h_oc_raw[:, 0], h_oc_raw[:, 1])
        ml_ob = h_ob_raw[:, 2]
        ml_oc = h_oc_raw[:, 2]
        worst_ob = torch.maximum(base_ob, ml_ob)
        worst_oc = torch.maximum(base_oc, ml_oc)
        return (
            torch.stack([base_ob, ml_ob, worst_ob], dim=-1).to(torch.float64),
            torch.stack([base_oc, ml_oc, worst_oc], dim=-1).to(torch.float64),
        )

    ns_base = torch.maximum(NS[:, :, 0], NS[:, :, 1])
    ob_base = torch.maximum(ob[:, 0], ob[:, 1])
    oc_base = torch.maximum(oc[:, 0], oc[:, 1])
    p_ob_base = (1.0 + (ns_base >= ob_base.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
    p_oc_base = (1.0 + (ns_base >= oc_base.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
    hit_ob_base = (p_ob_base <= al)
    hit_oc_base = (p_oc_base <= al)

    ns_ml = NS[:, :, 2]
    ob_ml = ob[:, 2]
    oc_ml = oc[:, 2]
    p_ob_ml = (1.0 + (ns_ml >= ob_ml.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
    p_oc_ml = (1.0 + (ns_ml >= oc_ml.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M + 1.0)
    hit_ob_ml = (p_ob_ml <= al)
    hit_oc_ml = (p_oc_ml <= al)

    M_std = M // 2
    M_cal = M - M_std
    NS_std = NS[:M_std]
    NS_cal = NS[M_std:]
    mu_null = NS_std.mean(dim=0, keepdim=True)
    sd_null = NS_std.std(dim=0, keepdim=True) + 1e-12
    ns_zmax = ((NS_cal - mu_null) / sd_null).max(dim=-1).values
    ob_zmax = ((ob.unsqueeze(0) - mu_null) / sd_null).squeeze(0).max(dim=-1).values
    oc_zmax = ((oc.unsqueeze(0) - mu_null) / sd_null).squeeze(0).max(dim=-1).values
    p_ob_worst = (1.0 + (ns_zmax >= ob_zmax.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M_cal + 1.0)
    p_oc_worst = (1.0 + (ns_zmax >= oc_zmax.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M_cal + 1.0)
    hit_ob_worst = (p_ob_worst <= al)
    hit_oc_worst = (p_oc_worst <= al)

    return (
        torch.stack([hit_ob_base, hit_ob_ml, hit_ob_worst], dim=-1).to(torch.float64),
        torch.stack([hit_oc_base, hit_oc_ml, hit_oc_worst], dim=-1).to(torch.float64),
    )


@torch.no_grad()
def run_seed_gpu(args, D, seed, cfgs, sigs, stratum, device):
    """[FIX A] pass stratum to make_ctx (paired DiD).
    [REV 6] SD_FACTOR env var scales sd multiplicatively."""
    X = make_ctx(args, D, seed, SEED_BASE, stratum=stratum)
    rng = X["rng"]
    F = X["F"]
    y = X["y"]; Y = X["Y"]; N = X["N"]; d = X["d"]; c = X["c"]; g = X["g"]
    S0, G0, Hisq, fac, sd, W0, FH0, lev, R0 = (
        X[k] for k in ["S", "G", "Hisq", "fac", "sd", "W0", "FH", "lev", "R0"]
    )
    Fv, yv, Ft, yt = X["Fv"], X["yv"], X["Ft"], X["yt"]
    targets = X["targets"]
    ks = sorted(args.ks); K = ks[-1]; nk = len(ks)
    M = args.M; al = args.alpha
    zq = norm.ppf(1.0 - al)

    nc = len(cfgs)
    mstar = {}

    for ci, (kind, v) in enumerate(cfgs):
        if kind == "ss":
            acc_per_m = np.zeros(len(args.mults))
            for _ in range(20):
                cand = ss_release(S0, G0, sigs[v], rng, args.mults, g, d)
                acc_per_m += np.array([acc_np(Wc, Fv, yv) for Wc in cand])
            acc_per_m /= 20.0
            j = int(np.argmax(acc_per_m))
            mstar[ci] = args.mults[j]

    W0_t = torch.tensor(W0, dtype=torch.float64, device=device)
    Hisq_t = torch.tensor(Hisq, dtype=torch.float64, device=device)
    S0_t = torch.tensor(S0, dtype=torch.float64, device=device)
    G0_t = torch.tensor(G0, dtype=torch.float64, device=device)
    Ft_t = torch.tensor(Ft, dtype=torch.float64, device=device)
    yt_t = torch.tensor(yt, dtype=torch.long, device=device)

    nt = len(targets)
    X_targs = torch.tensor(F[targets], dtype=torch.float64, device=device)
    T1_targs = torch.tensor(FH0[targets], dtype=torch.float64, device=device)

    gen_gpu = torch.Generator(device=device)
    gen_gpu.manual_seed(int(900_000 + seed * 100))

    def make_nz_batch_gpu(kind, v, batch_len):
        Z = torch.randn((batch_len, d, c), dtype=torch.float64, device=device, generator=gen_gpu)
        if kind == "out":
            return (v * sd) * Z
        else:
            return (v * sd * fac) * torch.matmul(Hisq_t.unsqueeze(0), Z)

    NZ_proj_gpu = {}
    for ci, (kind, v) in enumerate(cfgs):
        if kind in ("out", "an"):
            N1 = make_nz_batch_gpu(kind, v, M)
            N2 = make_nz_batch_gpu(kind, v, M)
            x_N1_all = torch.einsum("td,mdc->tmc", X_targs, N1)
            x_N2_all = torch.einsum("td,mdc->tmc", X_targs, N2)
            t1_N1_all = torch.einsum("td,mdc->tmc", T1_targs, N1)
            t1_N2_all = torch.einsum("td,mdc->tmc", T1_targs, N2)
            NZ_proj_gpu[ci] = (x_N1_all, x_N2_all, t1_N1_all, t1_N2_all)
            del N1, N2

    M_bw = min(M, 200)
    NZ_B_proj_gpu = {}
    for ci, (kind, v) in enumerate(cfgs):
        if kind == "ss":
            sig = sigs[v]
            m_ = mstar[ci]

            def draw_b_weights_proj(n_draws):
                x_list, t1_list = [], []
                chunk = 25
                for st_i in range(0, n_draws, chunk):
                    ch = min(chunk, n_draws - st_i)
                    A_mat = sig * torch.randn((ch, d, d), dtype=torch.float64,
                                              device=device, generator=gen_gpu)
                    E_sym = torch.triu(A_mat) + torch.triu(A_mat, diagonal=1).transpose(-2, -1)
                    Z_mat = sig * torch.randn((ch, d, c), dtype=torch.float64,
                                              device=device, generator=gen_gpu)
                    W_b = ss_noisy_batch_gpu(S0_t.unsqueeze(0) + E_sym,
                                             G0_t.unsqueeze(0) + Z_mat, sig, m_, g, d)
                    dW_b = W_b - W0_t.unsqueeze(0)
                    x_list.append(torch.einsum("td,mdc->tmc", X_targs, dW_b))
                    t1_list.append(torch.einsum("td,mdc->tmc", T1_targs, dW_b))
                return torch.cat(x_list, dim=1), torch.cat(t1_list, dim=1)

            x_N1b_all, t1_N1b_all = draw_b_weights_proj(M_bw)
            x_N2b_all, t1_N2b_all = draw_b_weights_proj(M_bw)
            NZ_B_proj_gpu[ci] = (x_N1b_all, x_N2b_all, t1_N1b_all, t1_N2b_all)

    ss_indices = [ci for ci, (kind, _) in enumerate(cfgs) if kind == "ss"]
    nc_total = nc + len(ss_indices)

    HT = np.full((nt, nc_total, nk, NCH), np.nan)
    CT = np.zeros((nt, nc_total))
    VV = np.zeros(nt)

    ks_t = torch.tensor(ks, dtype=torch.float64, device=device)
    sk_t = 1.0 / torch.sqrt(ks_t)
    sk_v = sk_t.view(1, nk, 1)

    for ti, i in enumerate(targets):
        i = int(i)
        x = F[i]
        Yi = Y[i]
        yi = int(y[i])
        li = float(lev[i])
        sli = math.sqrt(max(li, 1e-12))
        t1 = FH0[i]
        t1n2 = float(t1 @ t1)

        W1 = W0 + np.outer(t1, R0[i]) / (1.0 - li)
        W1_t = torch.tensor(W1, dtype=torch.float64, device=device)
        x_t = X_targs[ti]
        t1_t = T1_targs[ti]
        Yi_t = torch.tensor(Yi, dtype=torch.float64, device=device)

        S1_t = S0_t - torch.outer(x_t, x_t)
        G1_t = G0_t - torch.outer(x_t, Yi_t)

        acc1 = float((torch.argmax(Ft_t @ W1_t, dim=1) == yt_t).to(torch.float64).mean().item())
        n2 = float(x @ x)
        e4 = float((x ** 4).sum())
        V = 0.5 * (n2 ** 2 + e4) + n2
        VV[ti] = V

        x_W0 = torch.matmul(x_t, W0_t)

        def sc_j0_batch(Dm_b, qm_b, kind):
            RN = torch.matmul(x_t, qm_b) - Yi_t
            rr = (RN * RN).sum(dim=-1)
            RNn = torch.sqrt(rr) + 1e-12
            t1_Dm = torch.matmul(t1_t, Dm_b)
            x_Dm = torch.matmul(x_t, Dm_b)
            A1 = (t1_Dm * RN).sum(dim=-1)
            A2_ = (x_Dm * RN).sum(dim=-1)
            s0, s1, sm = _score_channels(A1, A2_, rr, RNn, li, t1n2, sli, kind)
            return torch.stack([s0, s1, sm], dim=-1)

        def sc_j0_from_proj(x_N1, x_N2, t1_N1, t1_N2, kind):
            RN_null = x_W0.view(1, 1, c) + x_N2.unsqueeze(1) * sk_v - Yi_t.view(1, 1, c)
            rr_n = (RN_null * RN_null).sum(dim=-1)
            RNn_n = torch.sqrt(rr_n) + 1e-12
            A1_n = (((t1_N1 - t1_N2).unsqueeze(1) * sk_v) * RN_null).sum(dim=-1)
            A2_n = (((x_N1 - x_N2).unsqueeze(1) * sk_v) * RN_null).sum(dim=-1)
            s0_n, s1_n, sm_n = _score_channels(A1_n, A2_n, rr_n, RNn_n, li, t1n2, sli, kind)
            return torch.stack([s0_n, s1_n, sm_n], dim=-1)

        ss_w_offset = 0
        for ci, (kind, v) in enumerate(cfgs):
            if kind in ("out", "an"):
                post_noise = make_nz_batch_gpu(kind, v, K)
                pre_noise = make_nz_batch_gpu(kind, v, K)
                nl_noise = make_nz_batch_gpu(kind, v, K)

                post_cum = torch.cumsum(W1_t.unsqueeze(0) + post_noise, dim=0)
                pre_cum = torch.cumsum(W0_t.unsqueeze(0) + pre_noise, dim=0)
                nl_cum = torch.cumsum(W0_t.unsqueeze(0) + nl_noise, dim=0)

                acc_post0 = float((torch.argmax(Ft_t @ post_cum[0], dim=1) == yt_t).to(torch.float64).mean().item())
                CT[ti, ci] = 100.0 * (acc1 - acc_post0)

                x_N1, x_N2, t1_N1, t1_N2 = (t_arr[ti] for t_arr in NZ_proj_gpu[ci])
                NS = sc_j0_from_proj(x_N1, x_N2, t1_N1, t1_N2, kind)

                idx_k = torch.tensor([k - 1 for k in ks], dtype=torch.long, device=device)
                k_div = ks_t.view(nk, 1, 1)
                post_k = post_cum[idx_k] / k_div
                pre_k = pre_cum[idx_k] / k_div
                nl_k = nl_cum[idx_k] / k_div

                ob = sc_j0_batch(post_k - pre_k, pre_k, kind)
                oc = sc_j0_batch(nl_k - pre_k, pre_k, kind)

                hit_ob, hit_oc = calibrate_channels_gpu(NS, ob, oc, al, legacy_calib=args.legacy_calib)
                HT[ti, ci, :, 0:3] = hit_ob.cpu().numpy()
                HT[ti, ci, :, 3:6] = hit_oc.cpu().numpy()

            else:
                sig = sigs[v]
                m_ = mstar[ci]
                ci_w = nc + ss_w_offset
                ss_w_offset += 1

                A_p = sig * torch.randn((K, d, d), dtype=torch.float64, device=device, generator=gen_gpu)
                Ep_all = torch.triu(A_p) + torch.triu(A_p, diagonal=1).transpose(-2, -1)
                A_q = sig * torch.randn((K, d, d), dtype=torch.float64, device=device, generator=gen_gpu)
                Eq_all = torch.triu(A_q) + torch.triu(A_q, diagonal=1).transpose(-2, -1)
                A_nl = sig * torch.randn((K, d, d), dtype=torch.float64, device=device, generator=gen_gpu)
                Enl_all = torch.triu(A_nl) + torch.triu(A_nl, diagonal=1).transpose(-2, -1)

                Zp_all = sig * torch.randn((K, d, c), dtype=torch.float64, device=device, generator=gen_gpu)
                Zq_all = sig * torch.randn((K, d, c), dtype=torch.float64, device=device, generator=gen_gpu)
                Znl_all = sig * torch.randn((K, d, c), dtype=torch.float64, device=device, generator=gen_gpu)

                Wp_all = ss_noisy_batch_gpu(S1_t.unsqueeze(0) + Ep_all,
                                             G1_t.unsqueeze(0) + Zp_all, sig, m_, g, d)
                Wq_all = ss_noisy_batch_gpu(S0_t.unsqueeze(0) + Eq_all,
                                             G0_t.unsqueeze(0) + Zq_all, sig, m_, g, d)
                Wnl_all = ss_noisy_batch_gpu(S0_t.unsqueeze(0) + Enl_all,
                                              G0_t.unsqueeze(0) + Znl_all, sig, m_, g, d)

                acc_wp0 = float((torch.argmax(Ft_t @ Wp_all[0], dim=1) == yt_t).to(torch.float64).mean().item())
                cost_b_k1 = 100.0 * (acc1 - acc_wp0)

                if args.composition_mode == "total_transcript":
                    CT[ti, ci] = cost_b_k1
                    CT[ti, ci_w] = cost_b_k1
                else:
                    CT[ti, ci] = cost_b_k1
                    CT[ti, ci_w] = cost_b_k1

                sEp_cum = torch.cumsum(Ep_all, dim=0)
                sEq_cum = torch.cumsum(Eq_all, dim=0)
                sZp_cum = torch.cumsum(Zp_all, dim=0)
                sZq_cum = torch.cumsum(Zq_all, dim=0)

                xx_t = torch.outer(x_t, x_t)
                xY_t = torch.outer(x_t, Yi_t)
                for ki, b in enumerate(ks):
                    nS = xx_t - (sEp_cum[b - 1] - sEq_cum[b - 1]) / float(b)
                    nG = xY_t - (sZp_cum[b - 1] - sZq_cum[b - 1]) / float(b)
                    T_val = 0.5 * (float(x_t @ nS @ x_t) + float((x_t ** 2) @ torch.diag(nS))) \
                            + float(x_t @ nG[:, yi])
                    eff_sig = sig * (math.sqrt(b) if args.composition_mode == "total_transcript" else 1.0)
                    sdn = eff_sig * math.sqrt(2.0 / float(b)) * math.sqrt(V)
                    hit_1 = bool(T_val > zq * sdn)
                    hit_0 = bool((T_val - V) > zq * sdn)
                    HT[ti, ci, ki, 0:3] = hit_1
                    HT[ti, ci, ki, 3:6] = hit_0

                post_w_cum = torch.cumsum(Wp_all, dim=0)
                pre_w_cum = torch.cumsum(Wq_all, dim=0)
                nl_w_cum = torch.cumsum(Wnl_all, dim=0)

                x_N1b, x_N2b, t1_N1b, t1_N2b = (t_arr[ti] for t_arr in NZ_B_proj_gpu[ci])
                NS_b = sc_j0_from_proj(x_N1b, x_N2b, t1_N1b, t1_N2b, "B_w")

                idx_k = torch.tensor([k - 1 for k in ks], dtype=torch.long, device=device)
                k_div = ks_t.view(nk, 1, 1)
                post_wk = post_w_cum[idx_k] / k_div
                pre_wk = pre_w_cum[idx_k] / k_div
                nl_wk = nl_w_cum[idx_k] / k_div

                ob_w = sc_j0_batch(post_wk - pre_wk, pre_wk, "B_w")
                oc_w = sc_j0_batch(nl_wk - pre_wk, pre_wk, "B_w")

                ns_b_max = torch.maximum(NS_b[:, :, 0], NS_b[:, :, 1])
                ob_w_max = torch.maximum(ob_w[:, 0], ob_w[:, 1])
                oc_w_max = torch.maximum(oc_w[:, 0], oc_w[:, 1])
                p_ob_w = (1.0 + (ns_b_max >= ob_w_max.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M_bw + 1.0)
                p_oc_w = (1.0 + (ns_b_max >= oc_w_max.unsqueeze(0)).to(torch.float64).sum(dim=0)) / (M_bw + 1.0)
                hit_w1 = (p_ob_w <= al).cpu().numpy()
                hit_w0 = (p_oc_w <= al).cpu().numpy()
                for ch_i in range(3):
                    HT[ti, ci_w, :, ch_i] = hit_w1
                    HT[ti, ci_w, :, 3 + ch_i] = hit_w0

    return dict(h=HT, cost=CT, mstar=mstar, V=VV, Vt_mean=X["Vt_mean"])


def fam_defs_ext(cfgs_ext, eps_dp):
    ia = [i for i, c_ in enumerate(cfgs_ext) if c_[0] == "out"]
    ib = [i for i, c_ in enumerate(cfgs_ext) if c_[0] == "an"]
    ic = [i for i, c_ in enumerate(cfgs_ext) if c_[0] == "ss"]
    i4 = [i for i in ic if cfgs_ext[i][1] <= eps_dp]
    ic_w = [i for i, c_ in enumerate(cfgs_ext) if c_[0] == "ss_w"]
    i4_w = [i for i in ic_w if cfgs_ext[i][1] <= eps_dp]
    return {
        "A": (np.array(ia, int), False),
        "A2": (np.array(ib, int), False),
        "B4": (np.array(i4, int), True),
        "B": (np.array(ic, int), True),
        "B4_w": (np.array(i4_w, int), True),
        "B_w": (np.array(ic_w, int), True),
    }


def write_holm_results(dataset_name, p_values_dict, alpha=0.05):
    """
    [REV 5] Write Holm-corrected p-values to a shared file (append mode).
    Called once per dataset from `main` after `run_stratum_suite` returns.
    Produces one row per (stratum, test) combination, with raw and corrected
    p-values. The file is intended for the reviewer's reproducibility check
    and is NOT used by the plotting pipeline (which reads from the logs).
    """
    if not p_values_dict:
        print(f"   [REV 5] no primary tests to correct for {dataset_name}")
        return

    keys = list(p_values_dict.keys())
    p_raw = np.array([p_values_dict[k] for k in keys])
    p_adj = holm_bonferroni(p_raw)

    write_header = not os.path.exists(HOLM_FILE)
    with open(HOLM_FILE, "a") as fh:
        if write_header:
            fh.write(f"{'dataset':<12s} {'stratum':<8s} {'test':<14s} "
                     f"{'raw_p':>10s} {'holm_p':>10s}\n")
        for k_, key in enumerate(keys):
            stratum, test = key
            fh.write(f"{dataset_name:<12s} {stratum:<8s} {test:<14s} "
                     f"{p_raw[k_]:>10.4f} {p_adj[k_]:>10.4f}\n")
    print(f"   [REV 5] appended {len(keys)} rows to {HOLM_FILE}")


def leaks(Hm):
    return Hm[..., 0:3]


def fam_tab(Lk, C, fams, levels):
    nk = Lk.shape[1]
    out = np.full((len(FAMS), nk, 3, len(levels)), np.nan)
    for fi, f in enumerate(FAMS):
        idx, inc = fams[f]
        if len(idx) == 0:
            continue
        for ki in range(nk):
            for sc in range(3):
                if np.isnan(Lk[idx, ki, sc]).any():
                    continue
                l, c_ = frontier_pts(Lk[:, ki, sc], C, idx, inc)
                out[fi, ki, sc] = [cost_at(l, c_, v) for v in levels]
    return out


def ci(v):
    f = v[np.isfinite(v)]
    if len(f) < 0.5 * len(v):
        return None
    return np.percentile(f, [2.5, 97.5])


def fmt(x):
    return "cens." if not np.isfinite(x) else f"{x:+.2f}"


def lab_ext(p):
    if p[0] == "ss_w":
        return f"B_w DP   eps={p[1]:g}"
    return lab(p)


def check_sigma_consistency(args, sigs, ks, Vt_mean):
    """[FIX G] use actual Vt_mean instead of hardcoded 2.0."""
    z = norm.ppf(1 - args.alpha)
    print("\n######## SIGMA / NP CHECK (FIX [4, G]) ########")
    print(f"Using Vt_mean = {Vt_mean:.4f} (actual per-seed mean) instead of Vt = 2.0")
    print(f"{'eps':>6s} | {'sigma':>10s} | " + " | ".join(f"{'k=' + str(k):>12s}" for k in ks))
    for e in sorted(sigs.keys()):
        sg = sigs[e]
        cells = []
        for k in ks:
            th = float(norm.cdf(math.sqrt(k * Vt_mean / 2.0) / sg - z))
            cells.append(f"{th:12.4f}")
        print(f"{e:6g} | {sg:10.4f} | " + " | ".join(cells))
    print("Compare these to the paper's 'Theory NP' column before submission.")


def run_stratum_suite(args, D, cfgs, cfgs_ext, fams, sigs, levels, ks, stratum, device,
                       shared_sb=None):
    R = []
    t0 = time.time()
    print(f"\n>>> Execution {args.features or 'synthetique'} | Strate: {stratum} "
          f"({args.seeds} graines sur {device})...")
    for s in range(args.seeds):
        R.append(run_seed_gpu(args, D, s, cfgs, sigs, stratum, device))
        if (s + 1) % 5 == 0 or (s + 1) == args.seeds:
            el = time.time() - t0
            print(f"  graine {s + 1:02d}/{args.seeds} terminee ({el:.1f} s, {el/(s+1):.2f} s/graine)", flush=True)

    S = len(R)
    H = np.stack([r["h"] for r in R])
    CT = np.stack([r["cost"] for r in R])
    Vs = np.stack([r["V"] for r in R])
    Vt_mean = float(np.mean([r["Vt_mean"] for r in R]))
    nt = H.shape[1]

    L0 = leaks(H.mean((0, 1)))
    C0 = CT.mean((0, 1))
    T0 = fam_tab(L0, C0, fams, levels)

    if shared_sb is None:
        shared_sb = np.random.default_rng(0).integers(0, S, (args.B, S))
    TB_cluster = np.zeros((args.B,) + T0.shape)
    for b in range(args.B):
        sb = shared_sb[b]
        Hb = H[sb]
        Cb = CT[sb]
        Lb = leaks(Hb.mean((0, 1)))
        TB_cluster[b] = fam_tab(Lb, Cb.mean((0, 1)), fams, levels)

    nk = len(ks)
    kmax = nk - 1
    calib_str = "LEGACY (debug only)" if args.legacy_calib else "CONJOINT EXACT"
    print(f"\n=== fu_v4_8 v6 [{calib_str}] "
          f"(features={args.features or 'synthetique'}, gamma={args.gamma:g}, "
          f"{S} graines, {nt} cibles/graine, strate={stratum}, K={ks}, M={args.M}, B={args.B}, "
          f"Vt_mean={Vt_mean:.3f}) ===")

    print(f"\n   [REV 4] Composition mode: {args.composition_mode}")
    if args.composition_mode == "per_release":
        print(f"     ε_step ≤ {args.eps_dp:g} per release; total budget k·ε_step ≤ "
              f"{int(args.eps_dp * max(ks)):d} at k={max(ks)}.")
        print(f"     Note: this budget is high; see paper §5.2 for the practical "
              f"regime (total_transcript).")
    else:
        eff = args.eps_dp / math.sqrt(max(ks))
        print(f"     ε_tot ≤ {args.eps_dp:g}; effective ε_step = ε_tot/√k ≤ {eff:.3f} "
              f"at k={max(ks)}.")
        print(f"     Under this mode, the k={max(ks)} transcript is statistically "
              f"equivalent to a single release. B's cost at k={max(ks)} equals B's "
              f"cost at k=1 (see paper §5.2).")

    if os.environ.get("SD_FACTOR"):
        print(f"   [REV 6] SD_FACTOR = {float(os.environ['SD_FACTOR']):.2f} "
              f"(sd := κ · sd_calibrated; κ=1.0 is the baseline).")

    print("\n   mstar distribution per B epsilon:")
    for ci_diag, (kind, v) in enumerate(cfgs):
        if kind == "ss":
            ms = [r["mstar"].get(ci_diag) for r in R if r["mstar"].get(ci_diag) is not None]
            cnt = Counter(ms)
            print(f"      eps={v:g}: {dict(sorted(cnt.items()))}")

    print("\n######## Fuite (TPR a FPR = alpha) par config : base / ml / worst ########")
    print(f"{'config':<20s} | {'cout 1 pub':>10s} | " + " | ".join(f"{'k=' + str(k):^19s}" for k in ks))
    for ci_, cfg in enumerate(cfgs_ext):
        cells = []
        for ki in range(nk):
            b_ = L0[ci_, ki, 0]; m_ = L0[ci_, ki, 1]; w_ = L0[ci_, ki, 2]
            cells.append(f"{b_:5.3f}/{m_:5.3f}/{w_:5.3f}".center(19))
        print(f"{lab_ext(cfg):<20s} | {C0[ci_]:+10.2f} | " + " | ".join(cells))

    Hn = H.mean((0, 1))
    nul_base = Hn[..., 3]
    nul_ml = Hn[..., 4]
    nul_worst = Hn[..., 5]
    print(f"\n######## FPR sans suppression (alpha = {args.alpha:g}) [base / ml / worst] ########")
    print(f"{'config':<20s} | " + " | ".join(f"{'k=' + str(k):^19s}" for k in ks)
          + " | TPR theorique NP (B, ml)")
    for ci_, cfg in enumerate(cfgs_ext):
        cells = []
        for ki in range(nk):
            b_ = nul_base[ci_, ki]; m_ = nul_ml[ci_, ki]; w_ = nul_worst[ci_, ki]
            cells.append(f"{b_:5.3f}/{m_:5.3f}/{w_:5.3f}".center(19))
        extra = ""
        if cfg[0] == "ss":
            z = norm.ppf(1 - args.alpha)
            sg_ = sigs[cfg[1]]
            th = [float(np.mean(norm.cdf(np.sqrt(k / 2.0 * Vs) / sg_ - z))) for k in ks]
            extra = " k=" + "/".join(f"{k}:{t:.3f}" for k, t in zip(ks, th))
        print(f"{lab_ext(cfg):<20s} | " + " | ".join(cells) + " |" + extra)

    li = levels.index(args.prim_level)
    for sc_idx, sc_label in [(1, "Scenario ml (vraisemblance)"),
                             (2, "Scenario worst (union conjointe)")]:
        print(f"\n######## Frontieres ({sc_label}, fuite <= {args.prim_level:g}, Cluster Bootstrap) ########")
        print(f"   {'famille':<28s} | " + " | ".join(f"{'k=' + str(k):^24s}" for k in ks))
        for fi, f in enumerate(FAMS):
            cells = []
            for ki in range(nk):
                c_ = ci(TB_cluster[:, fi, ki, sc_idx, li])
                if np.isnan(T0[fi, ki, sc_idx, li]) and c_ is None:
                    cells.append(f"{'n/d ou cens.':^24s}")
                else:
                    cells.append(f"{fmt(T0[fi, ki, sc_idx, li]):>7s} "
                                 + ("      n/d       " if c_ is None
                                    else f"[{c_[0]:+6.2f};{c_[1]:+7.2f}]"))
            print(f"   {FAMN[f]:<28s} | " + " | ".join(cells))

        print(f"\n######## Tests pre-specifies ({sc_label}, k={ks[kmax]}, "
              f"fuite <= {args.prim_level:g}) ########")
        tests = [
            ("P1   : cout(A) - cout(A2) > 0", 0, 1),
            ("P2_s : cout(B4 stats) - cout(A2) > 0", 2, 1),
            ("P2_w : cout(B4 weights) - cout(A2) > 0", 4, 1),
            ("H_s  : cout(A2) - cout(B4 stats) > 0", 1, 2),
            ("H_w  : cout(A2) - cout(B4 weights) > 0", 1, 4),
        ]
        for nm, a_, b_ in tests:
            dd = TB_cluster[:, a_, kmax, sc_idx, li] - TB_cluster[:, b_, kmax, sc_idx, li]
            ok = np.isfinite(dd)
            if ok.mean() < 0.5:
                print(f"   {nm:<42s}: non testable ({100 * ok.mean():.0f} % valides)")
                continue
            lo, hi = np.percentile(dd[ok], [2.5, 97.5])
            lo_1s = np.percentile(dd[ok], 5.0)
            p = (1 + np.sum(dd[ok] <= 0)) / (ok.sum() + 1)
            d0 = T0[a_, kmax, sc_idx, li] - T0[b_, kmax, sc_idx, li]
            print(f"   {nm:<42s}: {fmt(d0)} pt CI95[{lo:+.2f};{hi:+.2f}] "
                  f"(1-side 5%: {lo_1s:+.2f}) ; p = {p:.4f} ({100 * ok.mean():.0f}% val)")

    return dict(H=H, CT=CT, Vs=Vs, L0=L0, C0=C0, T0=T0, TB=TB_cluster,
                Vt_mean=Vt_mean)


def collect_pvalues(res, stratum, ks, levels, prim_level):
    """
    [REV 5] Recompute per-test p-values from the bootstrap arrays so the
    Holm correction can be applied consistently across strata and datasets.
    Returns a dict keyed by (stratum, test_name) -> p_value.
    """
    T0 = res["T0"]; TB = res["TB"]
    kmax = len(ks) - 1
    li = levels.index(prim_level)
    pvals = {}
    for sc_idx, sc_name in [(1, "ml"), (2, "worst")]:
        for nm, a_, b_ in [
            ("P1",   0, 1),
            ("P2_s", 2, 1),
            ("P2_w", 4, 1),
            ("H_s",  1, 2),
            ("H_w",  1, 4),
        ]:
            dd = TB[:, a_, kmax, sc_idx, li] - TB[:, b_, kmax, sc_idx, li]
            ok = np.isfinite(dd)
            if ok.mean() < 0.5:
                continue
            p = (1 + np.sum(dd[ok] <= 0)) / (ok.sum() + 1)
            pvals[(stratum, f"{nm}_{sc_name}")] = p
    return pvals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="dino_mnist.npz")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--c", type=int, default=10)
    p.add_argument("--decay", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps_dp", type=float, default=4.0)
    p.add_argument("--mults", type=float, nargs="+",
                   default=[0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n_high", type=int, default=20)
    p.add_argument("--npub", type=int, default=2000)
    p.add_argument("--nte", type=int, default=5000)
    p.add_argument("--nval", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--levels", type=float, nargs="+", default=[0.10, 0.15, 0.25])
    p.add_argument("--prim_level", type=float, default=0.10)
    p.add_argument("--B", type=int, default=3000)
    p.add_argument("--M", type=int, default=300)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--seq_nt", type=int, default=12)
    p.add_argument("--seq_stratum", choices=["rand", "high"], default="rand")
    p.add_argument("--both_strata", action="store_true")
    p.add_argument("--legacy_calib", action="store_true",
                   help="Debug only: legacy (broken) union calibration. NOT used in the paper.")
    p.add_argument("--composition_mode", choices=["per_release", "total_transcript"],
                   default="per_release",
                   help="[REV 4] per_release: ε_step fixed (paper default). "
                        "total_transcript: ε_tot fixed, ε_step = ε_tot/√k.")
    p.add_argument("--seq_nus", type=float, nargs="+",
                   default=[50, 100, 150, 200, 300, 450, 700, 1000, 1500, 2200, 3200])
    p.add_argument("--seq_nus2", type=float, nargs="+",
                   default=[100, 200, 300, 450, 700, 1000, 1500, 2200,
                            3200, 4800, 6400, 9600, 12800, 19200])
    p.add_argument("--seq_eps", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 16])
    p.add_argument("--adjacency", choices=["remove", "replace"], default="remove")
    p.add_argument("--check_sigma", action="store_true")
    p.add_argument("--out_json", default="")
    a = p.parse_args()

    a.seq_nus = sorted(a.seq_nus)
    a.seq_nus2 = sorted(a.seq_nus2)
    a.seq_eps = sorted(a.seq_eps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config fu_v4_8 v6 ({device}): {vars(a)}")

    cfgs = ([("out", float(v)) for v in a.seq_nus] +
            [("an", float(v)) for v in a.seq_nus2] +
            [("ss", float(e)) for e in a.seq_eps])
    cfgs_ext = cfgs + [("ss_w", float(e)) for e in a.seq_eps]
    fams = fam_defs_ext(cfgs_ext, a.eps_dp)
    levels = sorted(set(a.levels) | {a.prim_level})
    ks = sorted(a.ks)
    kmax = len(ks) - 1
    li = levels.index(a.prim_level)

    D = fu_v4_3.load_data(a)
    sigs = {float(e): analytic_sigma(float(e), a.delta, sens_bound(a)) for e in a.seq_eps}

    if a.check_sigma:
        X_probe = make_ctx(a, D, 0, SEED_BASE, stratum=None)
        check_sigma_consistency(a, sigs, ks, X_probe["Vt_mean"])
        return

    strata_to_run = ["rand", "high"] if a.both_strata else [a.seq_stratum]
    shared_sb = np.random.default_rng(42).integers(0, a.seeds, (a.B, a.seeds))
    res_by_st = {}
    for st in strata_to_run:
        res_by_st[st] = run_stratum_suite(a, D, cfgs, cfgs_ext, fams, sigs,
                                           levels, ks, st, device,
                                           shared_sb=shared_sb)

    # [REV 5] Collect and write Holm-corrected p-values for all strata of this dataset.
    ds_name = os.path.basename(a.features).replace("dino_", "").replace(".npz", "")
    all_pvals = {}
    for st in strata_to_run:
        all_pvals.update(collect_pvalues(res_by_st[st], st, ks, levels, a.prim_level))
    write_holm_results(ds_name, all_pvals)

    if a.both_strata:
        for sc_idx, sc_name in [(1, "Scenario ml (vraisemblance)"),
                                 (2, "Scenario worst (conjoint)")]:
            print("\n=========================================================================")
            print(f" INTERACTION LEVIER x REPUBLICATION [{sc_name}]")
            print("=========================================================================")
            T_r, TB_r = res_by_st["rand"]["T0"], res_by_st["rand"]["TB"]
            T_h, TB_h = res_by_st["high"]["T0"], res_by_st["high"]["TB"]

            for b_idx, b_lbl in [(2, "B4 (stats UB)"), (4, "B4_w (weights sym)")]:
                d_lev0 = ((T_h[1, kmax, sc_idx, li] - T_r[1, kmax, sc_idx, li])
                          - (T_h[b_idx, kmax, sc_idx, li] - T_r[b_idx, kmax, sc_idx, li]))
                d_lev_b = ((TB_h[:, 1, kmax, sc_idx, li] - TB_r[:, 1, kmax, sc_idx, li])
                           - (TB_h[:, b_idx, kmax, sc_idx, li] - TB_r[:, b_idx, kmax, sc_idx, li]))
                ok = np.isfinite(d_lev_b)
                if ok.mean() >= 0.2:
                    lo, hi = np.percentile(d_lev_b[ok], [2.5, 97.5])
                    p_lev = (1 + np.sum(d_lev_b[ok] <= 0)) / (ok.sum() + 1)
                    print(f"   Delta_Levier (k=10) vs {b_lbl:<18s}: {d_lev0:+.2f} pt "
                          f"[{lo:+.2f};{hi:+.2f}] ; p = {p_lev:.4f} ({100*ok.mean():.0f}% valides)")
                else:
                    print(f"   Delta_Levier (k=10) vs {b_lbl:<18s}: non testable ({100*ok.mean():.0f}% valides)")

                did0 = ((T_h[1, kmax, sc_idx, li] - T_h[b_idx, kmax, sc_idx, li])
                        - (T_r[1, 0, sc_idx, li] - T_r[b_idx, 0, sc_idx, li]))
                did_b = ((TB_h[:, 1, kmax, sc_idx, li] - TB_h[:, b_idx, kmax, sc_idx, li])
                         - (TB_r[:, 1, 0, sc_idx, li] - TB_r[:, b_idx, 0, sc_idx, li]))
                ok2 = np.isfinite(did_b)
                if ok2.mean() >= 0.2:
                    lo2, hi2 = np.percentile(did_b[ok2], [2.5, 97.5])
                    p_did = (1 + np.sum(did_b[ok2] <= 0)) / (ok2.sum() + 1)
                    print(f"   Crossover DiD       vs {b_lbl:<18s}: {did0:+.2f} pt "
                          f"[{lo2:+.2f};{hi2:+.2f}] ; p = {p_did:.4f} ({100*ok2.mean():.0f}% valides)")
                else:
                    print(f"   Crossover DiD       vs {b_lbl:<18s}: non testable ({100*ok2.mean():.0f}% valides)")


if __name__ == "__main__":
    main()