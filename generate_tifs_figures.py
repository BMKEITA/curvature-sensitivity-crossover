#!/usr/bin/env python3
"""
generate_tifs_figures.py (v2.1 — fixed layout & synchronized with verified v4.8 GPU run)
Generates the three publication-ready IEEE TIFS figures (PDF + PNG):
  1. fig1_crossover_trajectories.pdf
  2. fig2_pareto_frontiers_k10.pdf
  3. fig3_hypothesis_forest_plot.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# =============================================================================
# IEEE TIFS Publication Styling
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "A":    "#d62728",
    "A2":   "#ff7f0e",
    "B4":   "#1f77b4",
    "B4_w": "#2ca02c",
    "B_w":  "#17becf",
}
MARKERS = {"A": "s", "A2": "o", "B4": "^", "B4_w": "D", "B_w": "P"}
LABELS = {
    "A":    "Defense A (Iso Weights)",
    "A2":   "Defense A2 (Aniso Weights)",
    "B4":   "Defense B4 (DP $\\varepsilon_{\\mathrm{step}}\\leq 4$, Stats UB)",
    "B4_w": "Defense B4_w (DP $\\varepsilon_{\\mathrm{step}}\\leq 4$, Weights Sym)",
    "B_w":  "Defense B_w (DP all $\\varepsilon_{\\mathrm{step}}$, Weights Sym)",
}
KS = [1, 2, 5, 10]
DATASETS = ["MNIST", "CIFAR-10", "CIFAR-100"]
DATASET_TITLES = {
    "MNIST":     "DINOv2 + MNIST ($c=10$)",
    "CIFAR-10":  "DINOv2 + CIFAR-10 ($c=10$)",
    "CIFAR-100": "DINOv2 + CIFAR-100 ($c=100$)",
}

# Set to False so verified built-in arrays are never overwritten by older log files
USE_RAW_LOGS = False


# =============================================================================
# 1. Exact v4.8 Frontier Trajectories (Table 2, Scenario `ml`)
# =============================================================================
FRONTIERS_ML = {
    "MNIST": {
        "rand": {
            "A":    {"pts": [4.88,  17.05, 32.83, 51.25], "ci10": (46.73, 57.17)},
            "A2":   {"pts": [0.05,   0.11,  0.30,  1.67], "ci10": (0.45,  3.09)},
            "B4":   {"pts": [8.20,   9.32, 12.36, 11.64], "ci10": (10.36, 12.73)},
            "B4_w": {"pts": [5.96,   5.96,  7.92,  5.96], "ci10": (5.88,  6.05)},
        },
        "high": {
            "A":    {"pts": [30.00, 57.95, 68.27, 70.89], "ci10": (67.42, 76.05)},
            "A2":   {"pts": [0.73,   1.13,  2.48,  7.95], "ci10": (3.87, 13.07)},
            "B4":   {"pts": [7.66,   8.75,  9.46,  9.67], "ci10": (9.27, 10.26)},
            "B4_w": {"pts": [5.96,   5.96,  5.96,  5.96], "ci10": (5.88,  6.05)},
        },
    },
    "CIFAR-10": {
        "rand": {
            "A":    {"pts": [1.37,  3.02,  9.72, 14.87], "ci10": (10.21, 23.36)},
            "A2":   {"pts": [0.11,  0.15,  0.53,  1.28], "ci10": (0.50,  3.52)},
            "B4":   {"pts": [1.61,  2.05,  2.35,  2.57], "ci10": (2.11,  3.17)},
            "B4_w": {"pts": [1.18,  1.18,  1.18,  1.18], "ci10": (1.12,  1.24)},
        },
        "high": {
            "A":    {"pts": [4.10, 17.23, 23.23, 42.32], "ci10": (38.06, 48.10)},
            "A2":   {"pts": [0.54,  1.41,  2.63,  2.84], "ci10": (2.40,  6.24)},
            "B4":   {"pts": [1.50,  1.77,  2.17,  2.16], "ci10": (2.00,  2.88)},
            "B4_w": {"pts": [1.18,  1.18,  1.18,  1.18], "ci10": (1.12,  1.24)},
        },
    },
    "CIFAR-100": {
        "rand": {
            "A":    {"pts": [18.05, 29.08, 59.49, 63.39], "ci10": (56.74, 70.56)},
            "A2":   {"pts": [ 3.62,  5.55,  9.99, 20.93], "ci10": (12.68, 29.89)},
            "B4":   {"pts": [10.34, 12.11, 14.82, 19.03], "ci10": (15.47, 22.55)},
            "B4_w": {"pts": [ 6.90,  6.90, 13.07, 23.34], "ci10": (15.54, 31.54)},
        },
        "high": {
            "A":    {"pts": [26.59, 42.59, 60.96, 65.65], "ci10": (63.72, 71.34)},
            "A2":   {"pts": [ 5.23,  6.41, 16.53, 36.97], "ci10": (29.86, 45.18)},
            "B4":   {"pts": [11.51, 10.57, 19.97, 22.43], "ci10": (16.87, 28.41)},
            "B4_w": {"pts": [ 6.90,  9.78, 18.51, 23.86], "ci10": (16.38, 35.40)},
        },
    },
}


# =============================================================================
# 2. Pareto Fronts at k=10 on `high` (Cost vs. Realized TPR)
# =============================================================================
PARETO_HIGH_K10_DEFAULT = {
    "MNIST": {
        "A":    [(0.66, 1.000), (3.15, 0.967), (9.93, 0.850), (19.83, 0.725),
                 (40.11, 0.478), (59.14, 0.203), (71.36, 0.097), (77.66, 0.069),
                 (81.92, 0.069), (83.74, 0.047), (85.11, 0.047)],
        "A2":   [(0.01, 0.994), (0.05, 0.878), (0.11, 0.714), (0.26, 0.531),
                 (0.66, 0.344), (1.43, 0.203), (3.87, 0.125), (10.10, 0.089),
                 (22.92, 0.067), (42.36, 0.056), (55.24, 0.058), (68.41, 0.050),
                 (74.38, 0.042), (79.70, 0.053)],
        "B4":   [(5.96, 1.000), (7.85, 0.581), (10.79, 0.211), (14.96, 0.119),
                 (21.38, 0.069), (50.00, 0.044)],
        "B4_w": [(5.96, 0.081), (7.85, 0.056), (10.79, 0.039), (14.96, 0.058),
                 (21.38, 0.092)],
    },
    "CIFAR-10": {
        "A":    [(0.07, 1.000), (0.35, 0.944), (0.87, 0.844), (1.71, 0.658),
                 (4.56, 0.383), (12.59, 0.244), (30.23, 0.153), (48.00, 0.075),
                 (63.13, 0.089), (72.03, 0.067), (77.48, 0.050)],
        "A2":   [(0.01, 0.992), (0.03, 0.881), (0.08, 0.703), (0.22, 0.464),
                 (0.58, 0.297), (1.28, 0.175), (3.30, 0.083), (8.03, 0.086),
                 (18.03, 0.072), (35.44, 0.064), (48.47, 0.047), (63.45, 0.058),
                 (70.54, 0.050), (77.10, 0.069)],
        "B4":   [(1.50, 1.000), (2.00, 0.567), (3.00, 0.225), (5.00, 0.094),
                 (10.00, 0.047)],
        "B4_w": [(1.18, 0.056), (3.00, 0.075), (5.00, 0.069), (10.00, 0.042)],
    },
    "CIFAR-100": {
        "A":    [(0.51, 1.000), (2.09, 1.000), (4.68, 0.994), (8.37, 0.967),
                 (18.05, 0.797), (34.06, 0.533), (54.83, 0.242), (67.09, 0.089),
                 (74.25, 0.075), (77.11, 0.089), (78.45, 0.064)],
        "A2":   [(0.04, 1.000), (0.25, 1.000), (0.55, 0.969), (1.27, 0.850),
                 (3.12, 0.550), (6.41, 0.372), (14.00, 0.169), (26.66, 0.142),
                 (43.37, 0.081), (60.41, 0.067), (68.68, 0.081), (74.92, 0.075),
                 (77.05, 0.033), (78.43, 0.050)],
        "B4":   [(6.90, 0.603), (10.70, 0.222), (17.44, 0.125), (29.69, 0.072),
                 (50.33, 0.067)],
        "B4_w": [(6.90, 0.281), (10.70, 0.203), (17.44, 0.122), (29.69, 0.083),
                 (50.33, 0.078)],
    },
}


# =============================================================================
# 3. Forest Plot Data (Table 5 & Table 6, Scenario `ml` and `worst` + Holm p-values)
# =============================================================================
FOREST_DATA = {
    "H_tests": [
        # (Label, ml_est, ml_ci, ml_p, worst_est, worst_ci, worst_p, holm_str)
        ("MNIST\n(vs. B4 stats)",     -1.72, ( -5.62,  3.31), 0.7960, -6.05, ( -7.11, -2.98), 1.0000, "Holm p=0.980"),
        ("MNIST\n(vs. B4_w weights)", +1.99, ( -2.10,  7.12), 0.1080, -2.34, ( -3.18,  0.86), 0.9240, "Holm p=0.216"),
        ("CIFAR-10\n(vs. B4 stats)",  +0.67, ( +0.05,  3.85), 0.0210, +0.37, ( -0.89,  2.38), 0.2860, "Holm p=0.063"),
        ("CIFAR-10\n(vs. B4_w weights)", +1.66, ( +1.21, 5.04), 0.0003, +1.35, ( +0.04, 3.40), 0.0110, "Holm p<0.001***"),
        ("CIFAR-100\n(vs. B4 stats)", +1.90, ( -8.54, 12.91), 0.4900, -2.48, ( -8.95,  4.00), 0.7480, "Holm p=0.980"),
        ("CIFAR-100\n(vs. B4_w weights)", -2.41, (-16.24, 11.49), 0.6810, -6.79, (-15.68, 3.14), 0.9050, "Holm p=0.681"),
    ],
    "DiD_tests": [
        ("MNIST\n(vs. B4 stats)",     +6.43, ( +2.45, 11.41), 0.0007, +2.11, ( -0.12,  5.04), 0.0340, "Holm p=0.0014**"),
        ("MNIST\n(vs. B4_w weights)", +7.90, ( +3.82, 13.03), 0.0003, +3.58, ( +2.77,  6.77), 0.0003, "Holm p<0.001***"),
        ("CIFAR-10\n(vs. B4 stats)",  +2.17, ( +1.45,  5.39), 0.0007, +1.93, ( +0.60,  3.98), 0.0020, "Holm p=0.0014**"),
        ("CIFAR-10\n(vs. B4_w weights)", +2.73, ( +2.27, 6.15), 0.0003, +2.48, ( +1.17, 4.55), 0.0003, "Holm p<0.001***"),
        ("CIFAR-100\n(vs. B4 stats)", +21.26, (+12.10, 33.38), 0.0003, +8.85, ( -4.73, 31.42), 0.1340, "Holm p<0.001***"),
        ("CIFAR-100\n(vs. B4_w weights)", +16.40, ( +3.90, 27.16), 0.0070, +3.98, (-13.95, 23.24), 0.3280, "Holm p=0.007**"),
    ],
}


# =============================================================================
# Figure 1: 2x3 Crossover Trajectories
# =============================================================================
def plot_fig1_crossover_trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), sharex=True)
    x_pos = np.arange(len(KS))
    offsets = {"A": -0.09, "A2": -0.03, "B4": +0.03, "B4_w": +0.09}

    for row_idx, (stratum, row_label) in enumerate([
        ("rand", "Random Targets (rand)"),
        ("high", "High-Leverage Targets (high)"),
    ]):
        for col_idx, ds in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            d_sub = FRONTIERS_ML[ds][stratum]

            for fam in ["A", "A2", "B4", "B4_w"]:
                pts = d_sub[fam]["pts"]
                ci_lo, ci_hi = d_sub[fam]["ci10"]
                xp = x_pos + offsets[fam]

                ax.plot(
                    xp, pts,
                    marker=MARKERS[fam],
                    color=COLORS[fam],
                    linewidth=2.0 if fam in ("A2", "B4_w") else 1.6,
                    linestyle="-" if fam in ("A", "A2") else "--",
                    markersize=6,
                    label=LABELS[fam] if (row_idx == 0 and col_idx == 0) else None,
                    zorder=4 if fam in ("A2", "B4_w") else 3,
                )

                y10 = pts[-1]
                err_low = max(0.0, y10 - ci_lo)
                err_high = max(0.0, ci_hi - y10)
                ax.errorbar(
                    [xp[-1]], [y10],
                    yerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor=COLORS[fam],
                    elinewidth=1.4,
                    capsize=3.5,
                    alpha=0.85,
                    zorder=5,
                )

            if stratum == "high":
                a2_10 = d_sub["A2"]["pts"][-1]
                bw_10 = d_sub["B4_w"]["pts"][-1]
                b4_10 = d_sub["B4"]["pts"][-1]
                best_b = min(bw_10, b4_10)
                if a2_10 > best_b:
                    ax.annotate(
                        f"Crossover\nA2: +{a2_10:.2f} pt\nDP: +{best_b:.2f} pt",
                        xy=(x_pos[-1] + offsets["A2"], a2_10),
                        xytext=(x_pos[-2] - 0.35, a2_10 * 1.15 + 3.0),
                        fontsize=7.8,
                        fontweight="bold",
                        color="#2b2b2b",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fff8dc", ec="#ff7f0e", alpha=0.92),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#ff7f0e", lw=1.1),
                        zorder=6,
                    )

            if row_idx == 0:
                ax.set_title(DATASET_TITLES[ds], pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nAccuracy Drop Cost (pts) at TPR $\\leq 0.10$")
            if row_idx == 1:
                ax.set_xlabel("Number of Checkpoint Republications ($k$)")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"$k={k}$" for k in KS])
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(bottom=0.0, top=105.0)
            ax.set_yticks([0, 0.5, 1, 2, 5, 10, 25, 50, 80])
            ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "25", "50", "80"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig1_crossover_trajectories.pdf", bbox_inches="tight")
    fig.savefig("fig1_crossover_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig1_crossover_trajectories.pdf & .png")


# =============================================================================
# Figure 2: 1x3 Full Privacy-Utility Pareto Frontiers at k=10 (high)
# =============================================================================
def plot_fig2_pareto_frontiers_k10():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)

    for col_idx, ds in enumerate(DATASETS):
        ax = axes[col_idx]
        p_data = PARETO_HIGH_K10_DEFAULT[ds]

        ax.axhspan(0.0, 0.10, color="#e6f5ea", alpha=0.55, zorder=1)
        ax.axhline(0.10, color="#2ca02c", linestyle=":", linewidth=1.4, zorder=2)
        ax.axhline(0.05, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)

        for fam in ["A", "A2", "B4", "B4_w"]:
            curve = p_data[fam]
            xs = [max(pt[0], 0.008) for pt in curve]
            ys = [pt[1] for pt in curve]
            ax.plot(
                xs, ys,
                marker=MARKERS[fam],
                color=COLORS[fam],
                linewidth=2.0 if fam in ("A2", "B4_w") else 1.6,
                linestyle="-" if fam in ("A", "A2") else "--",
                markersize=5.5,
                label=LABELS[fam] if col_idx == 0 else None,
                zorder=4 if fam in ("A2", "B4_w") else 3,
            )

        ax.set_title(f"{DATASET_TITLES[ds]} — High Leverage ($k=10$)", pad=8)
        ax.set_xlabel("Single-Release Accuracy Drop Cost (pts, log scale)")
        ax.set_xscale("log")
        ax.set_xlim(left=0.008, right=110.0)
        ax.set_ylim(-0.02, 1.05)

        if col_idx == 0:
            ax.set_ylabel("Realized Unlearning Leakage $\\mathrm{TPR}$ ($\\alpha=0.05$)")
            ax.text(0.012, 0.115, "Target Privacy Threshold ($\\mathrm{TPR} \\leq 0.10$)", fontsize=7.8, color="#1b7837", fontweight="bold")
            ax.text(0.012, 0.012, "Nominal Null $\\mathrm{FPR} = 0.05$", fontsize=7.5, color="#555555")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig("fig2_pareto_frontiers_k10.pdf", bbox_inches="tight")
    fig.savefig("fig2_pareto_frontiers_k10.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig2_pareto_frontiers_k10.pdf & .png")


# =============================================================================
# Figure 3: 1x2 Cluster Bootstrap Forest Plot (H_s/H_w & DiD_s/DiD_w)
# =============================================================================
def plot_fig3_hypothesis_forest_plot():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True)

    panels = [
        ("H_tests", "Panel A: Direct High-Leverage Reversal at $k=10$\n$\\mathrm{Cost}_{\\mathrm{high}}(\\mathrm{A2}) - \\mathrm{Cost}_{\\mathrm{high}}(\\mathrm{B4}) > 0$ (pts)"),
        ("DiD_tests", "Panel B: Republication $\\times$ Leverage Crossover DiD\n$[\\mathrm{Cost}_{\\mathrm{high}}(\\mathrm{A2}) - \\mathrm{Cost}_{\\mathrm{high}}(\\mathrm{B4})]_{k=10} - [\\mathrm{Cost}_{\\mathrm{rand}}(\\mathrm{A2}) - \\mathrm{Cost}_{\\mathrm{rand}}(\\mathrm{B4})]_{k=1}$"),
    ]
    y_positions = np.arange(len(FOREST_DATA["H_tests"]))[::-1]

    for p_idx, (key, title) in enumerate(panels):
        ax = axes[p_idx]
        rows = FOREST_DATA[key]

        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.3, zorder=2)
        ax.axvspan(0.0, 68.0, color="#f0f9e8", alpha=0.45, zorder=1)

        for idx, (label, ml_est, ml_ci, ml_p, w_est, w_ci, w_p, holm_str) in enumerate(rows):
            y = y_positions[idx]

            # Primary Scenario `ml` (top offset)
            y_ml = y + 0.15
            ax.errorbar(
                [ml_est], [y_ml],
                xerr=[[max(0.0, ml_est - ml_ci[0])], [max(0.0, ml_ci[1] - ml_est)]],
                fmt="o",
                color="#1f77b4",
                ecolor="#1f77b4",
                elinewidth=1.8,
                capsize=3.5,
                markersize=5.8,
                zorder=4,
            )
            sig_ml = "*" if ml_p <= 0.05 else ""
            ax.text(
                ml_ci[1] + 0.8, y_ml,
                f"{ml_est:+.2f} pt ($p={ml_p:.4f}${sig_ml}; {holm_str})",
                va="center", fontsize=7.3, color="#1f77b4",
                fontweight="bold" if ml_p <= 0.05 else "normal",
            )

            # Jointly Calibrated Scenario `worst` (bottom offset)
            y_w = y - 0.15
            ax.errorbar(
                [w_est], [y_w],
                xerr=[[max(0.0, w_est - w_ci[0])], [max(0.0, w_ci[1] - w_est)]],
                fmt="s",
                color="#ff7f0e",
                ecolor="#ff7f0e",
                elinewidth=1.5,
                capsize=3.0,
                markersize=5.0,
                alpha=0.9,
                zorder=3,
            )
            sig_w = "*" if w_p <= 0.05 else ""
            ax.text(
                w_ci[1] + 0.8, y_w,
                f"{w_est:+.2f} pt ($p={w_p:.4f}${sig_w})",
                va="center", fontsize=7.1, color="#d95f02",
                fontweight="bold" if w_p <= 0.05 else "normal",
            )

        ax.set_title(title, pad=8, fontsize=9.8)
        ax.set_xlabel("Accuracy Cost Advantage in Favor of Defense B (percentage points)")
        ax.set_xlim(-19.5, 68.0)
        ax.set_yticks(y_positions)
        if p_idx == 0:
            ax.set_yticklabels([r[0] for r in rows], fontweight="bold")

    ml_handle = mlines.Line2D(
        [], [], color="#1f77b4", marker="o", linestyle="-",
        linewidth=1.8, markersize=6,
        label="Primary Calibrated Likelihood-Ratio Audit (Scenario ml + Holm adjustment)",
    )
    w_handle = mlines.Line2D(
        [], [], color="#ff7f0e", marker="s", linestyle="-",
        linewidth=1.5, markersize=5.2,
        label="Jointly Calibrated Multi-Channel Union (Scenario worst)",
    )
    zone_patch = mpatches.Patch(
        facecolor="#f0f9e8", edgecolor="#2ca02c", alpha=0.6,
        label="Region where Defense B Outperforms A2 ($\\Delta\\mathrm{Cost} > 0$)",
    )

    fig.legend(
        handles=[ml_handle, w_handle, zone_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig("fig3_hypothesis_forest_plot.pdf", bbox_inches="tight")
    fig.savefig("fig3_hypothesis_forest_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig3_hypothesis_forest_plot.pdf & .png")


if __name__ == "__main__":
    plot_fig1_crossover_trajectories()
    plot_fig2_pareto_frontiers_k10()
    plot_fig3_hypothesis_forest_plot()
    print("\nAll three publication-ready figures generated successfully.")
