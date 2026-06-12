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
  author  = {Liu, Yiming and Zhang, Wentao and Wang, Xutong and Wu, Guangxin},
  journal = {Advances in Manufacturing},
  year    = {2026},
  note    = {Under review}
}
```

## Companion Repository

CFD-DPM bucket search method for bubble-inclusion contact screening:  
[github.com/YimLiu626/-ladle-cfd-bucket-screening](https://github.com/YimLiu626/-ladle-cfd-bucket-screening)

The companion paper (under review at *Advances in Manufacturing*) describes the VOF-DPM framework used to generate the CFD reference dataset in this work — same ladle geometry, same five argon flow rates (40–120 NL/min).

## License

MIT License — see [LICENSE](LICENSE) file.

## Contact

Yiming Liu — School of Materials Science and Engineering, Shanghai University  
Guangxin Wu (corresponding) — gxwu@shu.edu.cn
