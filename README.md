# The Curvature–Sensitivity Crossover in Repeated Unlearning Republications

**Reproducibility Artifact for IEEE TIFS Submission**  
*Per-Target Calibrated Auditing of Heuristic Weight-Space and Differentially Private Sufficient-Statistic Defenses on Frozen Foundation-Model Ridge Heads*

---

## 1. Repository Structure & Contents

This repository contains the complete source code, shell launchers, statistical verification scripts, figure generators, and full CUDA execution logs required to reproduce every table (Tables 1–3) and figure (Figures 1–3) in the manuscript:

```text
├── fu_v4_3.py                        # Core CPU exact Sherman-Morrison unlearning & DP engine
├── fu_v4_8_gpu.py                    # Batched CUDA per-target Monte Carlo auditor (M=300, B=3000)
├── generate_tifs_figures.py          # Generates Fig. 1, Fig. 2, and Fig. 3 (PDF + 300 DPI PNG)
├── run_all_experiments.sh            # Master end-to-end reproduction & verification launcher
├── holm_correction_results.txt       # Holm-Bonferroni FWER & bootstrap CI verification output
├── logs/                             # Complete verified execution logs
│   ├── 00_check_sigma.log            # Analytic Gaussian DP sigma & sensitivity (sqrt(2)) verification
│   ├── 01_cpu_quick.log              # CPU engine sanity check (4 seeds, n=4000, d=128)
│   ├── 02_gpu_quick.log              # GPU calibration sanity check (4 seeds, B=100, M=50)
│   ├── 03_mnist.log                  # Primary Benchmark: DINOv2 + MNIST (30 seeds, B=3000, M=300)
│   ├── 03_cifar10.log                # Primary Benchmark: DINOv2 + CIFAR-10 (30 seeds, B=3000, M=300)
│   ├── 03_cifar100.log               # Primary Benchmark: DINOv2 + CIFAR-100 (30 seeds, B=3000, M=300)
│   ├── sens_gamma_0.001_cifar100.log # Regularization sensitivity ablation (gamma = 0.001, 10 seeds)
│   ├── sens_gamma_0.01_cifar100.log  # Regularization sensitivity baseline (gamma = 0.01, 10 seeds)
│   └── sens_gamma_0.1_cifar100.log   # Regularization sensitivity ablation (gamma = 0.1, 10 seeds)
└── README.md                         # This step-by-step reproduction guide
