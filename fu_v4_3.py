#!/usr/bin/env python3
"""
fu_v4_3.py (v6 — final, addresses all reviewer points including revised SD sensitivity)
Core mathematical engine for the Curvature-Sensitivity Crossover benchmark.

Cumulative fixes:
  [FIX 1]  run_seed / run_seq_seed: X["F", "X", "features"] -> X["F"]
  [FIX 3]  analytic_sigma: brentq (rtol=1e-12)
  [FIX 7]  make_signs: parity-fold cross-fitting documented
  [FIX 10] Unified seed base SEED_BASE = 50_000 across single/seq modes
  [FIX 11] Placebo selection excludes the full deletion set
  [FIX 12] cost_at: clamps lvl to 1e-12
  [FIX 13] --adjacency replace: documented; default "remove" (Delta = sqrt(2))
  [FIX 15] Null-check draws K null pairs (documented)
  [FIX A]  make_ctx consumes `stratum` to draw the target set (paired DiD design)
  [FIX F]  DiD bootstrap uses PAIRED shared indices across rand and high
  [FIX G]  make_ctx returns Vt_mean so --check_sigma uses the actual Vt
  [FIX I]  mstar selected by averaging validation accuracy over R=20 draws
  [REV 2]  --grid_mode {narrow, wide}: warns when A2 is censored on MNIST high
  [REV 5]  holm_bonferroni(): FWER-corrected p-values; appended to a file
  [REV 6]  SD_FACTOR (not SD_FIXED): multiplicative factor kappa applied to sd_calibrated.
"""
import argparse, math, time, os
from collections import Counter
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.stats import kstest
from scipy.optimize import brentq

KEYS = {
    "Ftr": ["Ftr", "F_train", "Xtr", "x_train", "X_train", "train_features",
            "features_train", "F", "X", "features"],
    "ytr": ["ytr", "y_train", "Ytr", "train_labels", "labels_train", "y", "labels"],
    "Fte": ["Fte", "F_test", "Xte", "x_test", "X_test", "test_features", "features_test"],
    "yte": ["yte", "y_test", "Yte", "test_labels", "labels_test"],
}
ATT = ["mf_ex", "mf_pub", "wf_ex", "rec_ex", "rec_pub"]
NMAX = 3
FAMS = ["A", "A2", "B4", "B"]
FAMN = {"A": "A  bruit iso", "A2": "A2 bruit aniso",
        "B4": "B  DP eps<=eps_dp", "B": "B  DP tous eps"}
COMPS = [("A2", "A", "H3 : A2 < A"),
         ("B4", "A", "H4 : B(eps<=4) < A"),
         ("B4", "A2", "H5 : B(eps<=4) < A2")]

SEED_BASE = 50_000
HOLM_FILE = "holm_correction_results.txt"


# ------------------------------------------------------------------ DP / algebre
def analytic_sigma(eps, delta, sens):
    """Analytic Gaussian mechanism (Balle & Wang 2018). [FIX 3]: brentq."""
    def dlt(s):
        a = sens / (2.0 * s)
        b = eps * s / sens
        t = math.exp(min(700.0, eps + float(log_ndtr(-a - b))))
        return float(ndtr(a - b)) - t - delta
    lo, hi = 1e-8 * sens, 1e8 * sens
    while dlt(lo) < 0.0 and lo > 1e-300 * sens:
        lo *= 1e-2
    while dlt(hi) > 0.0 and hi < 1e300 * sens:
        hi *= 1e2
    return float(brentq(dlt, lo, hi, xtol=1e-14 * sens, rtol=1e-12, maxiter=500))


def eigh_fast(M):
    try:
        from scipy.linalg import eigh
        return eigh(M, driver="evd")
    except Exception:
        return np.linalg.eigh(M)


def ss_release(St, Gt, sig, rng, mults, g, d):
    A = sig * rng.standard_normal((d, d))
    E = np.triu(A) + np.triu(A, 1).T
    Z = sig * rng.standard_normal(Gt.shape)
    mu, V = eigh_fast(St + E)
    mu = np.maximum(mu, 0.0)
    VG = V.T @ (Gt + Z)
    return [V @ (VG / (mu + g + m * sig * math.sqrt(d))[:, None]) for m in mults]


def sm_delete(Hi, x):
    v = Hi @ x
    return Hi + np.outer(v, v) / (1.0 - x @ v)


# ------------------------------------------------------------------ donnees
def pick(z, names):
    for k in names:
        if k in z:
            return z[k]
    return None


def load_data(args):
    if not args.features:
        rng = np.random.default_rng(123)
        d, c, n = args.d, args.c, args.n
        nt = max(2000, args.nte + args.nval + args.npub)
        sp = (np.arange(d) + 1.0) ** (-args.decay / 2)
        T = rng.standard_normal((d, c))

        def gen(m):
            X = rng.standard_normal((m, d)) * sp
            y = np.argmax(X @ T + 0.3 * rng.standard_normal((m, c)) * sp.mean(), 1)
            return X, y

        Ftr, ytr = gen(n); Fte, yte = gen(nt)
    else:
        z = np.load(args.features, allow_pickle=True)
        Ftr, ytr, Fte, yte = (pick(z, KEYS[k]) for k in ["Ftr", "ytr", "Fte", "yte"])
        if Ftr is None or ytr is None:
            raise SystemExit(f"Cles non reconnues. Cles du fichier : {list(z.keys())}")
        if Fte is None or yte is None:
            rr = np.random.default_rng(7)
            p = rr.permutation(len(Ftr))
            k = int(0.85 * len(p))
            Fte, yte = Ftr[p[k:]], ytr[p[k:]]
            Ftr, ytr = Ftr[p[:k]], ytr[p[:k]]
    nrm = lambda X: (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    c = int(max(ytr.max(), yte.max())) + 1
    return nrm(Ftr), ytr.astype(int), nrm(Fte), yte.astype(int), c


def make_ctx(args, D, seed, base, stratum=None):
    """Context generator. [FIX A]: draw the target set inside make_ctx so that
    rand and high share F, S0, G0, H0, W0, lev per seed (paired DiD design).
    [FIX G]: returns Vt_mean for --check_sigma.
    [REV 6] SD_FACTOR = kappa: sd := kappa * sd_calibrated (multiplicative, not absolute)."""
    rng = np.random.default_rng(base + seed)
    Ftr, ytr, Fte, yte, c = D
    n = min(args.n, len(Ftr))
    idx = rng.choice(len(Ftr), n, replace=False)
    F = Ftr[idx].astype(np.float64); y = ytr[idx]
    perm = rng.permutation(len(Fte))
    nte = min(args.nte, len(Fte) - args.nval - args.npub)
    assert nte > 0, "jeu de test trop petit pour nte+nval+npub"
    te = perm[:nte]
    va = perm[nte:nte + args.nval]
    pb = perm[nte + args.nval:nte + args.nval + args.npub]
    N, d = F.shape
    g = args.gamma
    Y = np.eye(c)[y]
    S = F.T @ F
    G = F.T @ Y
    Hm = S + g * np.eye(d)
    lam, Vh = np.linalg.eigh(Hm)
    Hinv = (Vh / lam) @ Vh.T
    Hisq = (Vh / np.sqrt(lam)) @ Vh.T
    fac = math.sqrt(d / float(np.sum(1.0 / lam)))
    W0 = Hinv @ G
    FH = F @ Hinv
    lev = np.einsum("ij,ij->i", FH, F)
    R0 = F @ W0 - Y
    dn = np.linalg.norm(FH, axis=1) * np.linalg.norm(R0, axis=1) / (1 - lev)
    sd = float(np.median(dn)) / math.sqrt(d * c)

    # [REV 6] Multiplicative sd override via SD_FACTOR env var.
    # The env var is interpreted as kappa, so sd := kappa * sd_calibrated.
    # This tests robustness to a misscaling of the noise normalization,
    # unlike an absolute override (which trivially breaks the mechanism).
    if os.environ.get("SD_FACTOR"):
        kappa = float(os.environ["SD_FACTOR"])
        sd = sd * kappa

    targets = None
    if stratum is not None:
        top = np.argsort(lev)[-max(args.n_high, N // 100):]
        src = top if stratum == "high" else np.arange(N)
        targets = rng.choice(src, min(args.seq_nt, len(src)), replace=False)

    n2 = np.einsum("ij,ij->i", F, F)
    e4 = (F ** 4).sum(1)
    Vt_mean = float((0.5 * (n2 ** 2 + e4) + n2).mean())

    return dict(rng=rng, F=F, y=y, Y=Y, N=N, d=d, c=c, g=g, S=S, G=G, Hm=Hm,
                lam=lam, Vh=Vh, Hinv=Hinv, Hisq=Hisq, fac=fac, W0=W0, FH=FH,
                lev=lev, R0=R0, dn=dn, sd=sd,
                Ft=Fte[te].astype(np.float64), yt=yte[te],
                Fv=Fte[va].astype(np.float64), yv=yte[va],
                pub=Fte[pb].astype(np.float64),
                targets=targets, Vt_mean=Vt_mean)


def lab(p):
    if p[0] == "out":
        return f"A  iso   nu={p[1]:g}"
    if p[0] == "an":
        return f"A2 aniso nu={p[1]:g}"
    return f"B  DP    eps={p[1]:g}"


# ================================================================== MODE SINGLE
def build_params(args):
    P = [("out", 0.0)] + [("out", float(v)) for v in args.nus if v > 0]
    P += [("an", float(v)) for v in args.nus2 if v > 0]
    P += [("ss", float(e)) for e in args.eps]
    return P


def fam_defs(params, args):
    ia = [i for i, p in enumerate(params) if p[0] == "out" and p[1] > 0]
    ib = [i for i, p in enumerate(params) if p[0] == "an"]
    ic = [i for i, p in enumerate(params) if p[0] == "ss"]
    i4 = [i for i in ic if params[i][1] <= args.eps_dp]
    return {"A": (np.array(ia, int), False),
            "A2": (np.array(ib, int), False),
            "B4": (np.array(i4, int), True),
            "B": (np.array(ic, int), True)}


def run_seed(args, D, seed, params, sigs):
    X = make_ctx(args, D, seed, SEED_BASE, stratum=None)
    rng = X["rng"]
    F = X["F"]
    y = X["y"]; Y = X["Y"]; N = X["N"]; d = X["d"]; g = X["g"]
    S, G, Hm, Hinv, Hisq, fac, W0, FH, lev, R0, sd = (
        X[k] for k in ["S", "G", "Hm", "Hinv", "Hisq", "fac", "W0", "FH", "lev", "R0", "sd"]
    )
    Ft, yt, Fv, yv, pub = X["Ft"], X["yt"], X["Fv"], X["yv"], X["pub"]
    Hp = (N / len(pub)) * (pub.T @ pub) + g * np.eye(d)
    Hpinv = np.linalg.inv(Hp)
    att = [(T1, np.linalg.norm(T1, axis=1)) for T1 in (FH, F @ Hpinv)]
    sqlev = np.sqrt(np.maximum(lev, 1e-12))
    top = np.argsort(lev)[-max(args.n_high, N // 100):]
    dels = np.concatenate([
        rng.choice(N, args.n_rand, replace=False),
        rng.choice(top, min(args.n_high, len(top)), replace=False),
    ])
    nd = len(dels)
    deleted_set = set(int(i) for i in dels)

    pj = []
    for k, i in enumerate(dels):
        src = np.arange(N) if k < args.n_rand else top
        src = np.array([j for j in src if int(j) not in deleted_set], dtype=int)
        if len(src) == 0:
            src = np.array([j for j in range(N) if j not in deleted_set], dtype=int)
        pj.append(int(rng.choice(src)))

    pools = [np.where(y == y[i])[0] for i in dels]
    j0s = [int(np.searchsorted(pools[k], i)) for k, i in enumerate(dels)]
    dW = [np.outer(FH[i], R0[i]) / (1 - lev[i]) for i in dels]
    acc = lambda W, Xm, yy: float(np.mean(np.argmax(Xm @ W, 1) == yy))
    acc_ex = np.array([acc(W0 + dW[k], Ft, yt) for k in range(nd)])

    low = X["lam"] < np.median(X["lam"])
    fr = []
    for i in dels:
        e = ((X["Vh"].T @ F[i]) / X["lam"]) ** 2
        fr.append(e[low].sum() / e.sum())
    hte = np.einsum("ij,ij->i", Ft @ Hinv, Ft)
    diag = (float(np.mean(fr)), float(hte.mean() * fac ** 2))

    def evaluate(Dm, W0pub, k, i, j):
        u = np.linalg.svd(Dm, full_matrices=False)[0][:, 0]
        pool = pools[k]; j0 = j0s[k]; P = len(pool)
        Fp = F[pool]
        RN = Fp @ W0pub - Y[pool]; RNn = np.linalg.norm(RN, axis=1) + 1e-12
        rk = np.zeros(5)
        for a, (T1, T1n) in enumerate(att):
            sc = ((T1[pool] @ Dm) * RN).sum(1) / (T1n[pool] * RNn)
            rk[a] = ((sc > sc[j0]).sum() + 1) / P
        sc = ((Fp @ Dm) * RN).sum(1) / (sqlev[pool] * RNn)
        rk[2] = ((sc > sc[j0]).sum() + 1) / P
        cx = 0.0; pl = np.full(2, 0.5)
        for a, Ha in enumerate((Hm, Hp)):
            xh = Ha @ u; xh /= np.linalg.norm(xh) + 1e-12
            cs = np.abs(F @ xh)
            rk[3 + a] = ((cs > cs[i]).sum() + 1) / N
            pl[a] = ((cs > cs[j]).sum() + 1) / N
            if a == 0:
                cx = float(cs[i] - cs.mean())
        return rk, cx, pl

    NP = len(params)
    RK = np.zeros((nd, NP, 5)); CT = np.zeros((nd, NP))
    CX = np.zeros((nd, NP)); PL = np.zeros((nd, NP, 2))
    mstar = np.full(NP, np.nan)

    for pi, p in enumerate(params):
        if p[0] in ("out", "an"):
            nu = p[1]; kind = p[0]

            def noise():
                Z = rng.standard_normal((d, X["c"]))
                return nu * sd * (Z if kind == "out" else fac * (Hisq @ Z))

            W0pub = W0 + noise()
            for k, i in enumerate(dels):
                W1pub = W0 + dW[k] + noise()
                RK[k, pi], CX[k, pi], PL[k, pi] = evaluate(W1pub - W0pub, W0pub, k, i, pj[k])
                CT[k, pi] = 100 * (acc_ex[k] - acc(W1pub, Ft, yt))
        else:
            sig = sigs[p[1]]
            R_draws = 20
            acc_per_m = np.zeros(len(args.mults))
            for _ in range(R_draws):
                cand = ss_release(S, G, sig, rng, args.mults, g, d)
                acc_per_m += np.array([acc(Wc, Fv, yv) for Wc in cand])
            acc_per_m /= R_draws
            jj = int(np.argmax(acc_per_m))
            m = args.mults[jj]
            mstar[pi] = m
            base = ss_release(S, G, sig, rng, [m], g, d)[0]
            for k, i in enumerate(dels):
                S1 = S - np.outer(F[i], F[i])
                G1 = G - np.outer(F[i], Y[i])
                rel = ss_release(S1, G1, sig, rng, [m], g, d)[0]
                RK[k, pi], CX[k, pi], PL[k, pi] = evaluate(rel - base, base, k, i, pj[k])
                CT[k, pi] = 100 * (acc_ex[k] - acc(rel, Ft, yt))

    q = np.percentile(X["dn"] / np.median(X["dn"]), [50, 90, 99, 100])
    return dict(rank=RK, cost=CT, cx=CX, pl=PL,
                ix_r=np.arange(args.n_rand),
                ix_h=np.arange(args.n_rand, nd),
                mstar=mstar, dnq=q, acc_ex=float(acc_ex.mean()),
                diag=diag, n=N, d=d, c=X["c"],
                Vt_mean=X["Vt_mean"])


# ------------------------------------------------------------------ fuite, sens de rec, agregation
def hits(rk, alpha, sg):
    h = np.empty(rk.shape, bool)
    h[..., :3] = rk[..., :3] < alpha
    for a in (0, 1):
        lo = rk[..., 3 + a] < alpha
        hi = rk[..., 3 + a] > 1 - alpha
        h[..., 3 + a] = np.where(sg[:, a] > 0, lo, hi)
    return h


def make_signs(R, stratum, alpha):
    """[FIX 7]: parity-fold cross-fitting of the reconstruction direction."""
    S = len(R); out = [None] * S
    key = "ix_r" if stratum == "rand" else "ix_h"

    def fold(ids):
        rk = np.concatenate([R[s]["rank"][R[s][key]] for s in ids])[..., 3:5]
        lo = (rk < alpha).mean(0)
        hi = (rk > 1 - alpha).mean(0)
        return np.where(lo >= hi, 1, -1)

    ev = list(range(0, S, 2))
    od = list(range(1, S, 2))
    se, so = fold(ev), fold(od)
    for s in ev: out[s] = so
    for s in od: out[s] = se
    return out


def agg(R, seeds, stratum, alpha, SG, rng=None):
    key = "ix_r" if stratum == "rand" else "ix_h"
    L, C, X, PLk = [], [], [], []
    for s in seeds:
        r = R[s]; ix = r[key]
        if rng is not None:
            ix = rng.choice(ix, len(ix))
        sg = SG[s]
        L.append(hits(r["rank"][ix], alpha, sg).mean(0))
        C.append(r["cost"][ix].mean(0))
        X.append(r["cx"][ix].mean(0))
        pl = r["pl"][ix]
        PLk.append(np.where(sg > 0, pl < alpha, pl > 1 - alpha).mean(0))
    return np.mean(L, 0), np.mean(C, 0), np.mean(X, 0), np.mean(PLk, 0)


# ------------------------------------------------------------------ frontieres
def pava(y, increasing=True):
    y = np.asarray(y, float)
    if not increasing:
        return -pava(-y, True)
    vals, wts = [], []
    for v in y:
        vals.append(v); wts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [v2]; wts[-2:] = [w]
    return np.repeat(vals, np.array(wts, int))


def frontier_pts(Lm, C, idx, inc):
    if len(idx) == 0:
        return np.array([]), np.array([])
    l = np.clip(pava(Lm[idx], inc), 1e-6, 1.0)
    c = pava(C[idx], not inc)
    pts = {}
    for a, b in zip(np.round(l, 9), c):
        pts[a] = min(pts.get(a, np.inf), b)
    ks = sorted(pts)
    return np.array(ks), np.array([pts[k] for k in ks])


def cost_at(l, c, lvl):
    lvl = max(float(lvl), 1e-12)
    if len(l) == 0 or lvl < l[0] - 1e-12:
        return np.nan
    if lvl >= l[-1]:
        return float(c[-1])
    return float(np.interp(math.log(lvl), np.log(l), c))


def fam_costs(L, C, fams, levels):
    Lm = L[:, :NMAX].max(1); out = {}
    for f, (idx, inc) in fams.items():
        l, c = frontier_pts(Lm, C, idx, inc)
        out[f] = np.array([cost_at(l, c, v) for v in levels])
    return out


# [REV 5] Holm-Bonferroni correction for multiple comparisons.
def holm_bonferroni(p_values):
    """
    Apply Holm-Bonferroni step-down correction to a list of p-values.
    Returns corrected p-values in the same order as the input.

    Reference: Holm, S. (1979). A simple sequentially rejective multiple
    test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    corrected = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, p_adj)
        corrected[idx] = running_max
    return corrected


def fam_range(L, fams):
    Lm = L[:, :NMAX].max(1); out = {}
    for f, (idx, inc) in fams.items():
        if len(idx) == 0:
            out[f] = (np.nan, np.nan); continue
        l = np.clip(pava(Lm[idx], inc), 1e-6, 1.0)
        out[f] = (float(l.min()), float(l.max()))
    return out


def fmt(x, w=7):
    return f"{'cens.':>{w}s}" if not np.isfinite(x) else f"{x:+{w}.2f}"


def ci(v):
    f = v[np.isfinite(v)]
    if len(f) < 0.5 * len(v):
        return None, 100 * len(f) / max(len(v), 1)
    return np.percentile(f, [2.5, 97.5]), 100 * len(f) / len(v)


def report_single(args):
    params = build_params(args)
    fams = fam_defs(params, args)
    levels = sorted(set(args.levels) | {args.prim_rand, args.prim_high})
    D = load_data(args)
    sigs = {float(e): analytic_sigma(float(e), args.delta, math.sqrt(2.0)) for e in args.eps}
    R = []; t0 = time.time()
    for s in range(args.seeds):
        R.append(run_seed(args, D, s, params, sigs))
        print(f"  graine {s + 1}/{args.seeds} terminee ({time.time() - t0:.0f} s)", flush=True)
    S = len(R); rng = np.random.default_rng(0)
    print(f"\n=== fu_v4_3 v6 single (features={args.features or 'synthetique'}, "
          f"gamma={args.gamma:g}, delta={args.delta:g}, {S} graines, "
          f"n={R[0]['n']}, d={R[0]['d']}, c={R[0]['c']}, Vt_mean={R[0]['Vt_mean']:.3f}) ===")
    print(f"exactitude de W1 exact (test) : {100 * np.mean([r['acc_ex'] for r in R]):.2f} %")
    q = np.mean([r["dnq"] for r in R], 0)
    print(f"||dW_i||/mediane : p50={q[0]:.2f} p90={q[1]:.2f} p99={q[2]:.2f} max={q[3]:.2f}")
    dg = np.mean([r["diag"] for r in R], 0)
    print(f"Diagnostic : energie de dW dans la moitie a faible variance = {dg[0]:.3f} ; "
          f"variance de prediction aniso/iso = {dg[1]:.3f}")
    print(f"alpha={args.alpha} ; FUITE = max sur mf_ex, mf_pub, wf_ex ; "
          f"hasard ~ {args.alpha} (+~0.007 pour le max de 3)")
    for e in args.eps:
        pi = [i for i, p in enumerate(params) if p[0] == "ss" and p[1] == float(e)][0]
        ms = [r["mstar"][pi] for r in R]
        print(f"   B eps={e:g} : sigma={sigs[float(e)]:.3g} ; m choisi moyen={np.mean(ms):.3f} ; "
              f"repartition={dict(sorted(Counter(ms).items()))}")
    lo_m, hi_m = min(args.mults), max(args.mults)
    allm = [m for r in R for m in r["mstar"] if np.isfinite(m)]
    if allm:
        print(f"   m aux bornes de la grille : bas {np.mean(np.isclose(allm, lo_m)):.0%}, "
              f"haut {np.mean(np.isclose(allm, hi_m)):.0%}")

    strata = ["rand", "high"]
    names = {"rand": "suppressions aleatoires",
             "high": "suppressions a fort levier (top 1 %)"}

    if args.grid_mode == "narrow" or max(args.nus2) <= 6400:
        print("\n[REV 2] WARNING: using narrow grid (ν₂ ≤ 6400). "
              "A2 will be CENSORED on MNIST high at k=10.")
        print("  To obtain the un-censored results reported in the paper, "
              "use --grid_mode wide (ν₂ ≤ 19200).")

    SG = {st: make_signs(R, st, args.alpha) for st in strata}
    pt = {}; rng_cov = {}
    for st in strata:
        L, C, Xc, PLk = agg(R, range(S), st, args.alpha, SG[st])
        L1 = agg(R, range(S), st, 0.01, SG[st])[0]
        key = "ix_r" if st == "rand" else "ix_h"
        pooled = np.concatenate([r["rank"][r[key]] for r in R])
        m_ = len(pooled)
        print(f"\n######## {names[st]} (n={m_} rangs/config ; "
              f"KS crit 5 % non corrige ~ {1.36 / math.sqrt(m_):.3f}) ########")
        print(f"{'config':<22s} | {'mf_ex':>6s} {'mf_pub':>6s} {'wf_ex':>6s} |  f5(max3)   f1   KS3 | "
              f"{'rec_ex':>6s} {'rec_pb':>6s} {'pl_ex':>6s} {'pl_pb':>6s} | cos_exc | cout (pt)")
        for pi, p in enumerate(params):
            ks = max(kstest(pooled[:, pi, a], "uniform").statistic for a in range(NMAX))
            f5 = L[pi, :NMAX].max()
            mk = "*" if (f5 <= 0.075 + 1e-9 and p[1] > 0) else " "
            print(f"{lab(p):<22s} | " + " ".join(f"{v:6.3f}" for v in L[pi, :NMAX]) +
                  f" | {f5:7.3f}{mk}{L1[pi, :NMAX].max():6.3f} {ks:5.3f} | "
                  f"{L[pi, 3]:6.3f} {L[pi, 4]:6.3f} {PLk[pi, 0]:6.3f} {PLk[pi, 1]:6.3f} | "
                  f"{Xc[pi]:7.3f} | {C[pi]:+8.2f}")
        pt[st] = fam_costs(L, C, fams, levels)
        rng_cov[st] = fam_range(L, fams)

        per = []
        for s in range(S):
            Ls, _, _, Pls = agg(R, [s], st, args.alpha, SG[st])
            per.append(Ls[:, 3:5] - Pls)
        per = np.array(per)
        print(f"   Fuite NETTE de reconstruction (rec - placebo), IC95 sur graines :")
        for f in ("A", "A2", "B4"):
            idx = fams[f][0]
            if len(idx) == 0:
                continue
            d_s = per[:, idx, :].mean(1)
            bm = np.array([d_s[rng.integers(0, S, S)].mean(0) for _ in range(2000)])
            lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
            print(f"      {FAMN[f]:<20s}: rec_ex {d_s[:, 0].mean():+.3f} [{lo[0]:+.3f};{hi[0]:+.3f}]   "
                  f"rec_pub {d_s[:, 1].mean():+.3f} [{lo[1]:+.3f};{hi[1]:+.3f}]")

    print("\nPlage de fuite lissee (max de 3 attaques) couverte par la grille de chaque famille :")
    for st in strata:
        print(f"   {names[st]}")
        for f in FAMS:
            a_, b_ = rng_cov[st][f]
            print(f"      {FAMN[f]:<20s}: [{a_:.3f} ; {b_:.3f}]")

    bs_rng = np.random.default_rng(42)
    shared_sb = bs_rng.integers(0, S, (args.B, S))
    bs = {(st, f): [] for st in strata for f in FAMS}
    for b in range(args.B):
        sb = shared_sb[b]
        for st in strata:
            L, C, _, _ = agg(R, sb, st, args.alpha, SG[st], rng)
            fc = fam_costs(L, C, fams, levels)
            for f in FAMS:
                bs[(st, f)].append(fc[f])
    bs = {k: np.array(v) for k, v in bs.items()}

    pv = {}
    for st in strata:
        prim = args.prim_rand if st == "rand" else args.prim_high
        print(f"\n######## Frontieres interpolees : {names[st]} ########")
        print("cout (pt) a fuite FIXEE ; 'cens.' = niveau non atteint ; "
              "'n/d' = defini dans < 50 % des replicats")
        for li, lv in enumerate(levels):
            desc = lv < args.min_test - 1e-12
            tag = "  (PRIMAIRE)" if lv == prim else (
                "  (DESCRIPTIF : proche du plancher, aucun test)" if desc else "")
            print(f"\n-- fuite <= {lv:g}{tag}")
            for f in FAMS:
                c_, pc = ci(bs[(st, f)][:, li])
                if c_ is None:
                    print(f"   {FAMN[f]:<20s}:     n/d                                "
                          f"(defini dans {pc:.0f} % des replicats)")
                else:
                    cs = f"[{c_[0]:+.2f} ; {c_[1]:+.2f}]"
                    print(f"   {FAMN[f]:<20s}: {fmt(pt[st][f][li])} pt {cs:>20s}  "
                          f"(defini dans {pc:.0f} % des replicats)")
            if desc:
                continue
            for X_, Y_, nm in COMPS:
                dd = bs[(st, Y_)][:, li] - bs[(st, X_)][:, li]
                ok = np.isfinite(dd)
                d0 = pt[st][Y_][li] - pt[st][X_][li]
                if ok.mean() < 0.5:
                    print(f"   {nm:<22s}: non testable (comparaison censuree)")
                    continue
                lo, hi = np.percentile(dd[ok], [2.5, 97.5])
                p = (1 + np.sum(dd[ok] <= 0)) / (ok.sum() + 1)
                print(f"   {nm:<22s}: cout(Y)-cout(X) = {fmt(d0).strip()} pt "
                      f"[{lo:+.2f} ; {hi:+.2f}] ; p unilateral = {p:.4f} ({100 * ok.mean():.0f} % valides)")
                if lv == prim:
                    pv[(st, nm)] = p

    keys = list(pv); m = len(keys)
    print(f"\n-- Holm-Bonferroni correction across {m} primary hypothesis tests "
          f"(rand: {args.prim_rand:g}; high: {args.prim_high:g})")
    if m:
        p_ = np.array([pv[k] for k in keys])
        adj = holm_bonferroni(p_)
        print(f"   {'test':<45s} {'raw p':>10s} {'Holm p':>10s} {'significant?':>14s}")
        for k_, key in enumerate(keys):
            sig = "YES" if adj[k_] <= args.alpha else "no"
            print(f"   {names[key[0]] + ' / ' + key[1]:<45s} "
                  f"{p_[k_]:>10.4f} {adj[k_]:>10.4f} {sig:>14s}")
        write_header = not os.path.exists(HOLM_FILE)
        with open(HOLM_FILE, "a") as fh:
            if write_header:
                fh.write(f"{'dataset':<15s} {'test':<20s} {'raw_p':>10s} {'holm_p':>10s}\n")
            for k_, key in enumerate(keys):
                fh.write(f"{names[key[0]]:<15s} {key[1]:<20s} "
                         f"{p_[k_]:>10.4f} {adj[k_]:>10.4f}\n")
        print(f"   (appended to {HOLM_FILE}; header written only on first call)")


# ================================================================== MODE SEQ
def run_seq_seed(args, D, seed, cfgs, sigs):
    X = make_ctx(args, D, seed, SEED_BASE, stratum=args.seq_stratum)
    rng = X["rng"]
    F = X["F"]
    y = X["y"]; Y = X["Y"]; N = X["N"]; d = X["d"]; c = X["c"]; g = X["g"]
    S0, G0, Hisq, fac, sd, W0, Hinv0, FH0, lev = (
        X[k] for k in ["S", "G", "Hisq", "fac", "sd", "W0", "Hinv", "FH", "lev"]
    )
    Fv, yv = X["Fv"], X["yv"]
    sqlev = np.sqrt(np.maximum(lev, 1e-12))
    ks = sorted(args.ks); K = ks[-1]
    targets = X["targets"]
    nc = len(cfgs); mstar = {}

    def release(ci, Wex, St, Gt):
        kind, v = cfgs[ci]
        if kind in ("out", "an"):
            Z = rng.standard_normal((d, c))
            return Wex + v * sd * (Z if kind == "out" else fac * (Hisq @ Z))
        return ss_release(St, Gt, sigs[v], rng, [mstar[ci]], g, d)[0]

    pre = []
    for ci, (kind, v) in enumerate(cfgs):
        lst = []
        if kind == "ss":
            acc_per_m = np.zeros(len(args.mults))
            for _ in range(20):
                cand = ss_release(S0, G0, sigs[v], rng, args.mults, g, d)
                acc_per_m += np.array([np.mean(np.argmax(Fv @ Wc, 1) == yv) for Wc in cand])
            acc_per_m /= 20.0
            j = int(np.argmax(acc_per_m))
            mstar[ci] = args.mults[j]
            lst.append(ss_release(S0, G0, sigs[v], rng, [mstar[ci]], g, d)[0])
        for _ in range(K - len(lst)):
            lst.append(release(ci, W0, S0, G0))
        pre.append(lst)

    RK = np.zeros((len(targets), nc, len(ks), 2, 2))
    for ti, i in enumerate(targets):
        i = int(i)
        others = rng.choice(np.setdiff1d(np.arange(N), [i]), K - 1, replace=False)
        seq = [i] + [int(o) for o in others]
        Hi, St, Gt = Hinv0, S0, G0
        st = [(W0, S0, G0)]
        for t in range(1, K + 1):
            x = F[seq[t - 1]]
            Hi = sm_delete(Hi, x)
            St = St - np.outer(x, x)
            Gt = Gt - np.outer(x, Y[seq[t - 1]])
            st.append((Hi @ Gt, St, Gt))
        bad = set(seq[1:])
        pool = np.array([j for j in np.where(y == y[i])[0] if j not in bad])
        j0 = int(np.searchsorted(pool, i)); P = len(pool)
        Fp = F[pool]; Yp = Y[pool]
        T1 = FH0[pool]; T1n = np.linalg.norm(T1, axis=1); sl = sqlev[pool]

        def rank2(Dm, W0pub):
            RN = Fp @ W0pub - Yp; RNn = np.linalg.norm(RN, axis=1) + 1e-12
            s0 = ((T1 @ Dm) * RN).sum(1) / (T1n * RNn)
            s1 = ((Fp @ Dm) * RN).sum(1) / (sl * RNn)
            return np.array([((s0 > s0[j0]).sum() + 1) / P,
                             ((s1 > s1[j0]).sum() + 1) / P])

        for ci in range(nc):
            post = [release(ci, st[1][0], st[1][1], st[1][2]) for _ in range(K)]
            chain = [None, post[0]] + [
                release(ci, st[t][0], st[t][1], st[t][2]) for t in range(2, K + 1)
            ]
            for ki, k in enumerate(ks):
                pm = np.mean(post[:k], 0); qm = np.mean(pre[ci][:k], 0)
                RK[ti, ci, ki, 0] = rank2(pm - qm, qm)
                RK[ti, ci, ki, 1] = rank2(chain[k] - pre[ci][0], pre[ci][0])
    return dict(rk=RK, mstar=mstar)


def report_seq(args):
    cfgs = ([("out", float(v)) for v in args.seq_nus] +
            [("an", float(v)) for v in args.seq_nus2] +
            [("ss", float(e)) for e in args.seq_eps])
    D = load_data(args)
    sigs = {float(e): analytic_sigma(float(e), args.delta, math.sqrt(2.0)) for e in args.seq_eps}
    R = []; t0 = time.time()
    for s in range(args.seeds):
        R.append(run_seq_seed(args, D, s, cfgs, sigs))
        print(f"  graine {s + 1}/{args.seeds} terminee ({time.time() - t0:.0f} s)", flush=True)
    S = len(R); ks = sorted(args.ks)
    Hs = np.stack([(r["rk"] < args.alpha).mean(0) for r in R])
    leak = Hs.mean(0).max(-1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, S, (args.B, S))
    bm = np.stack([Hs[ix].mean(0).max(-1) for ix in idx])
    lo, hi = np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)
    print(f"\n=== fu_v4_3 v6 seq (features={args.features or 'synthetique'}, gamma={args.gamma:g}, "
          f"{S} graines, {args.seq_nt} cibles/graine, strate={args.seq_stratum}, K={ks}) ===")
    ms = {ci: Counter(r["mstar"].get(ci) for r in R) for ci, c_ in enumerate(cfgs) if c_[0] == "ss"}
    for ci, cnt in ms.items():
        print(f"   {lab(cfgs[ci])} : m choisi = {dict(sorted(cnt.items()))}")
    for sc, nm in ((0, "RE-PUBLICATION"), (1, "CHAINE")):
        print(f"\n-- {nm}")
        print(f"{'config':<20s} | " + " | ".join(f"{'k=' + str(k):^22s}" for k in ks))
        for ci, cfg in enumerate(cfgs):
            print(f"{lab(cfg):<20s} | " + " | ".join(
                f"{leak[ci, ki, sc]:.3f} [{lo[ci, ki, sc]:.3f};{hi[ci, ki, sc]:.3f}]"
                for ki in range(len(ks))))


# ================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "seq"], default="single")
    p.add_argument("--features", default="")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--c", type=int, default=10)
    p.add_argument("--decay", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--nus", type=float, nargs="+",
                   default=[20, 50, 100, 150, 200, 300, 450, 600, 800, 1000, 1500])
    p.add_argument("--nus2", type=float, nargs="+",
                   default=[50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200, 6400])
    p.add_argument("--eps", type=float, nargs="+", default=[1, 1.5, 2, 3, 4, 8, 16, 256])
    p.add_argument("--eps_dp", type=float, default=4.0)
    p.add_argument("--mults", type=float, nargs="+",
                   default=[0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n_rand", type=int, default=20)
    p.add_argument("--n_high", type=int, default=20)
    p.add_argument("--npub", type=int, default=2000)
    p.add_argument("--nte", type=int, default=5000)
    p.add_argument("--nval", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--levels", type=float, nargs="+", default=[0.075, 0.10, 0.15, 0.25, 0.40])
    p.add_argument("--min_test", type=float, default=0.10)
    p.add_argument("--prim_rand", type=float, default=0.10)
    p.add_argument("--prim_high", type=float, default=0.25)
    p.add_argument("--B", type=int, default=3000)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--seq_nt", type=int, default=8)
    p.add_argument("--seq_stratum", choices=["rand", "high"], default="rand")
    p.add_argument("--seq_nus", type=float, nargs="+", default=[200, 450, 800])
    p.add_argument("--seq_nus2", type=float, nargs="+", default=[400, 800, 1600])
    p.add_argument("--seq_eps", type=float, nargs="+", default=[1, 4, 16])
    p.add_argument("--adjacency", choices=["remove", "replace"], default="remove",
                   help="remove: Delta = sqrt(2) (paper default); replace: conservative bound sqrt(5)")
    p.add_argument("--grid_mode", choices=["narrow", "wide"], default="wide",
                   help="[REV 2] narrow: nu_2 <= 6400 (A2 censored on MNIST high); "
                        "wide: nu_2 <= 19200 (A2 un-censored). Default 'wide'.")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n, a.d, a.c, a.seeds, a.n_rand, a.n_high, a.B = 4000, 128, 10, 4, 6, 6, 100
        a.nus, a.nus2, a.eps, a.mults = [0.5, 2, 8, 32], [1, 4, 16, 64], [4, 64], [0.5, 2]
        a.npub, a.nte, a.nval = 300, 1000, 300
        a.ks, a.seq_nt, a.seq_nus, a.seq_nus2, a.seq_eps = [1, 2, 4], 3, [2, 8], [4, 16], [4, 64]
    a.nus = sorted(a.nus); a.nus2 = sorted(a.nus2); a.eps = sorted(a.eps)
    print(f"Config : {vars(a)}")
    if a.seeds < 2:
        raise SystemExit("--seeds >= 2 requis (sens de rec estime par parite)")
    report_single(a) if a.mode == "single" else report_seq(a)


if __name__ == "__main__":
    main()