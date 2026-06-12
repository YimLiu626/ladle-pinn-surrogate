# Ladle-PINN-Surrogate

**A CFD-Informed Surrogate for Rapid 3D Flow Prediction in Argon-Stirred Ladle Furnaces — VOF-Based Domain Decomposition, Cross-Position Generalization & Multi-Baseline Benchmarking**

[![DOI](https://img.shields.io/badge/Paper-Under%20Review-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Abstract

Repeated CFD simulations of argon-stirred ladle flow are computationally prohibitive. This work develops a CFD-informed surrogate combining volume-of-fluid (VOF)-based domain decomposition with neural network regression. An 8×384 SiLU MLP trained on VOF-filtered steel-phase data achieves **R² = 0.983** for center-blowing with near-zero extrapolation degradation. A binary injection-position parameter *m* ∈ {0,1} enables cross-position generalization (**R² = 0.94–0.97** for eccentric blowing). Six baselines are compared; POD-ROM achieves the highest center-blowing accuracy (**R² = 0.994**) but collapses across injection positions (**R² < 0**). The surrogate predicts the full 3D field in **~5 seconds** per condition (vs. 36 hours CFD) on consumer GPU hardware.

---

## Key Results

| Model | Center R² | Ecc. R² | Inference |
|-------|-----------|---------|-----------|
| **8×384 MLP (Ours)** | **0.983** | **0.94–0.97** | ~5 s |
| POD-ROM | 0.994 | < 0 (fails) | ~12 s |
| MLP 6×128 | 0.904 | — | ~3 s |
| PINN + continuity | 0.873 | — | ~5 s |
| DeepONet | −0.72 | — | ~5 s |
| Kriging (GPR) | Diverged | — | OOM |
| KAN | — | — | Toolchain |

## Repository Structure

```
├── src/                        # All source code
│   ├── config.py               # Global hyperparameter configuration
│   ├── data_loader.py          # CFD CSV → normalized tensors, VOF filtering, z-score normalization
│   ├── model.py                # PINN/MLP network (6×128, 8×384, SiLU activation)
│   ├── physics.py              # RANS residuals: continuity, momentum, mixing-length ν_t = l_m²|S|
│   ├── physics_scaled.py       # Physical-space gradient scaling for properly normalized physics loss
│   ├── train_pinn.py           # Two-phase training: Phase 1 (data-only) → Phase 2 (+physics +BC)
│   ├── full_train.py           # Main pipeline: pure data + λ_bc sweep + per-Q evaluation
│   ├── run_sweep.py            # λ_phys × λ_bc hyperparameter sweep (GPU-optimized)
│   ├── run_ablation.py         # ν_t ablation, physics-term ablation, VOF strategy ablation
│   ├── run_baselines.py        # DeepONet implementation, Kriging GPR, POD-ROM
│   ├── run_experiments.py      # Multi-seed stability, best model training, per-Q predictions
│   ├── complete_ablations.py   # Comprehensive ablation: physics, VOF, λ sweep, convergence
│   ├── final_ablations.py      # Full RANS: ν_t + l_m sensitivity + spatial decay
│   ├── finalize.py             # Final 8×384 SiLU MLP training (3000 epochs)
│   ├── inference.py            # Model inference, R² evaluation, denormalization
│   ├── slag_eye_mc.py          # Slag eye detection (α_slag < 0.5) + MC dropout uncertainty
│   └── plot/
│       ├── plot_field.py       # 2D CFD velocity contour maps + VOF phase distribution
│       ├── plot_3d.py          # 3D plume core scatter + steel volume rendering
│       ├── plot_paper_figures.py  # All 11 paper figures (600 DPI)
│       └── plot_4x3_error.py   # 4×3 PINN vs POD 3D error comparison
│
├── paper/                      # Manuscript
│   ├── main.tex                # Full LaTeX source
│   ├── main_final.pdf          # Compiled PDF (16 pages)
│   └── figures/                # Figure PDFs for LaTeX compilation
│
├── results/                    # Experiment outputs (JSON)
│   ├── final_results.json      # Best model: R², per-field scores, multi-seed stats
│   ├── cross_position.json     # m=0 blind vs. m∈{0,1} joint results
│   ├── baselines.json          # DeepONet, MLP, POD comparison
│   ├── complete_ablations.json # Physics, VOF, λ sweep, convergence curves
│   ├── pod_rom.json            # POD-ROM: R², modes, computational cost
│   └── ...
│
├── pic/                        # Final paper figures (600 DPI PNG)
│   ├── Fig1.png                # Framework schematic (AI-generated)
│   ├── Fig2.png – Fig11.png    # All paper figures
│   ├── figure_captions.txt     # Complete figure captions (11 figures)
│   ├── fig1_drawing_prompt.txt # AI drawing prompt used for Fig1
│   └── paper_text.txt          # Paper text (plain, extracted from LaTeX)
│
├── sample_q40.csv              # Sample CFD data: 100 rows × 12 columns
├── .gitignore
└── README.md
```

## Data Format

CFD data is exported from ANSYS Fluent as 12-column CSV (cell centers, all fluid zones):

| Column | Description | Units |
|--------|-------------|-------|
| cellnumber | Cell index | — |
| x-coordinate | X position | m |
| y-coordinate | Y position | m |
| z-coordinate | Z position | m |
| pressure | Static pressure | Pa |
| x-velocity | X velocity component | m/s |
| y-velocity | Y velocity component | m/s |
| z-velocity | Z velocity component | m/s |
| argon-vof | Argon volume fraction | [0,1] |
| steel-vof | Steel volume fraction | [0,1] |
| slag-vof | Slag volume fraction | [0,1] |
| air-vof | Air volume fraction (residual) | [0,1] |

VOF sum ≡ 1 verified for all 8 CFD cases. See `sample_q40.csv` for the exact format.

## Quick Start

### Environment

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn
```

GPU recommended (RTX 5070 Laptop 8GB used in paper). CPU training feasible with `6×128` model.

### Training

```bash
# Pure data baseline (8×384 SiLU MLP, 2500 epochs)
python src/full_train.py

# BC weight sweep
python src/run_sweep.py

# Full ablation suite
python src/complete_ablations.py
```

### Baselines

```bash
# DeepONet, POD-ROM, Kriging GPR
python src/run_baselines.py
```

### Inference

```bash
# Evaluate a trained model on all test conditions
python src/inference.py
```

## Project Explanation

### Problem Setting

Argon-stirred ladle furnace (LF) refining is a critical secondary metallurgy process. During bottom argon injection, buoyancy-driven bubble plumes generate strong recirculation in the molten steel bath, enhancing mixing and inclusion transport toward the slag layer. CFD simulation using VOF + DPM is the standard tool for studying this system, but a single steady-state case requires ~36 hours. When multiple gas flow rates (Q) and nozzle positions must be evaluated, repeated CFD becomes prohibitive.

### Approach

This work develops a **CFD-informed surrogate model** that learns to predict the full 3D velocity field from operating parameters alone — replacing expensive CFD with a feed-forward neural network that executes in ~5 seconds per condition.

**The key insight** is that the three-phase (argon/slag/steel) system spans four orders of magnitude in density, making full-domain physics losses ill-conditioned. In pure-steel cells (α_steel → 1), the VOF mixture equations reduce rigorously to single-phase incompressible RANS. By simply filtering training data to α_steel > 0.01, the problem is restricted to the region where physics is well-defined, without requiring explicit interface boundary conditions.

### Methods

**Architecture**: An 8×384 SiLU-activated multilayer perceptron maps (x, y, z, Q, m) → (u, v, w, p), where m ∈ {0,1} encodes the injection position (center vs. eccentric bottom nozzle). A 6×128 variant is used for ablation studies to reduce computational cost.

**Training**: Two-phase strategy — 500 epochs data-only, followed by up to 2500 epochs with optional continuity constraint. Adam optimizer (lr=10⁻³), batch size 8192, z-score normalization per output channel. Training completes in ~5 minutes on an RTX 5070 8GB GPU.

**Baselines**: Six methods compared — POD-ROM (SVD spatial modes + RBF coefficient interpolation), DeepONet (branch/trunk operator learning), Kriging GPR (sparse Gaussian process with 2000 inducing points), KAN (Kolmogorov-Arnold Networks), and two MLP variants (6×128, 8×384).

### Experiments

Eleven experiments systematically evaluate the surrogate:

1. **λ sensitivity sweep**: Continuity weight robust over [10⁻⁵, 10⁻¹]; BC weight provides zero benefit
2. **ν_t ablation**: Mixing-length, constant, and laminar ν_t all yield identical results — momentum residual does not help
3. **Physics term ablation**: Continuity adds +0.014 R² for the 6×128 model; BC and momentum add nothing
4. **Baseline comparison**: POD-ROM (0.994) > 8×384 MLP (0.983) > 6×128 MLP (0.904) > PINN+cont (0.873) > DeepONet (−0.72)
5. **Cross-position generalization**: Center-trained model collapses on eccentric data (R² < 0); joint m∈{0,1} training recovers R² = 0.94–0.97
6. **Convergence analysis**: Smooth convergence; errors concentrated in high-gradient plume core
7. **Extrapolation**: Near-zero degradation from Q=80 (training max) to Q=120
8. **VOF ablation**: Soft filtering (+0.344 R² over hard mask) is the single most impactful design choice
9. **l_m sensitivity**: No measurable effect (0.005–0.05 m)
10. **Multi-seed stability**: 0.978 ± 0.001 for 8×384 joint model
11. **Slag eye detection**: α_slag < 0.5 criterion at z = 1.85 m; 60.2% exposed area at Q=120 NL/min

### Key Findings

1. **VOF filtering is the dominant design factor.** Restricting training to steel-phase cells accounts for +0.344 R² — far more than physics constraints or architecture choices.

2. **POD-ROM is the best center-blowing baseline, but cannot cross positions.** Two SVD modes capture all snapshot variance from three training Q values; thin-plate-spline interpolation achieves R² = 0.994. However, POD spatial modes are configuration-specific and fail on eccentric data (R² < 0). Unlike the MLP, POD offers no mechanism for categorical input variables. Each nozzle position requires an independent POD model — the proposed surrogate replaces two with one.

3. **Physics regularization offers diminishing returns at scale.** The 6×128 model benefits from continuity (+0.014 R²); the 8×384 model does not. Momentum residuals never help because DPM bubble buoyancy — the actual plume driver — is absent from the homogeneous RANS formulation used in the PINN.

4. **Extrapolation is not a problem for this problem class.** Q-dependent flow variation is sufficiently smooth that three training points support accurate prediction at +50% beyond the training range.

5. **Parameterization is more powerful than physics.** The binary m variable — trivially implemented as an additional input channel — enables cross-position generalization that POD fundamentally cannot achieve, and does so without requiring physics-informed training.

### Reproducibility

All experiments are deterministic (fixed random seeds). The complete experiment pipeline — from data loading through training to figure generation — is self-contained in the `src/` directory. Result JSON files in `results/` contain exact numerical values for all tables and figures in the paper.

---

## Key Design Decisions

1. **VOF filtering > soft weighting**: Restricting to α_steel > 0.01 accounts for +0.344 R² improvement. Soft weighting adds negligible marginal benefit after filtering.

2. **Z-score normalization is critical**: Max-absolute scaling collapses pressure variance to near-zero (p ~ 133 kPa everywhere). Per-channel z-score normalization solves this.

3. **Physics regularization is marginal at scale**: Continuity adds +0.014 R² for 6×128 but zero gain for 8×384. Momentum and BC constraints do not help — the DPM-driven flow field is already encoded in CFD data.

4. **POD excels where it applies, fails where it doesn't**: 2-mode POD achieves R² = 0.994 on center-blowing (confirming low-rank Q-dependence) but collapses on eccentric data because spatial modes are configuration-specific.

## Citation

```bibtex
@article{liu2026surrogate,
  title   = {A CFD-Informed Surrogate for Rapid Flow Prediction in an Argon-Stirred Ladle:
             VOF-Based Domain Decomposition, Cross-Position Generalization, and Multi-Baseline Benchmarking},
  author  = {Liu, Y. and Zhang, W. and Wang, X. and Wu, G.},
  journal = {Advances in Manufacturing},
  year    = {2026},
  note    = {Under review}
}
```

## Model Zoo

Pre-trained models, inference demo, and benchmark scripts are available in the companion repository:  
[**PINN Model Zoo**](https://github.com/YimLiu626/pinn-model-zoo) — ready-to-use `.pt` weights, single-command inference, and unified benchmark across all test cases.

## Companion Repositories

CFD-DPM bucket search method for bubble-inclusion contact screening:  
[github.com/YimLiu626/-ladle-cfd-bucket-screening](https://github.com/YimLiu626/-ladle-cfd-bucket-screening)

The companion paper (under review at *Advances in Manufacturing*) describes the VOF-DPM framework used to generate the CFD reference dataset in this work — same ladle geometry, same five argon flow rates (40–120 NL/min).

## License

MIT License — see [LICENSE](LICENSE) file.

## Contact

School of Materials Science and Engineering, Shanghai University  
State Key Laboratory of Advanced Special Steel, Shanghai 200444, PR China
