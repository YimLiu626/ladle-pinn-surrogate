# Model Card: Ladle Flow Surrogate Models

## Model Inventory

| File | Architecture | Training | R² Center | R² Ecc. | Size |
|------|-------------|----------|-----------|----------|------|
| `center_8x384.pt` | 8×384 SiLU MLP | Center Q40,60,80 | 0.983 | — | 4.0 MB |
| `m0_8x384.pt` | 8×384 SiLU MLP | Center Q40,60,80 (m=0 only) | 0.980 | fails | 4.0 MB |
| `joint_8x384.pt` | 8×384 SiLU MLP | Center + Ecc. (m∈{0,1}) | 0.974 | 0.94–0.97 | 4.0 MB |

## Input/Output Specification

**Input** (5 channels): `(x, y, z, Q, m)`
- x, y, z: spatial coordinates [−1, 1], linearly mapped from domain bounds
- Q: argon flow rate [−1, 1], mapped as 2×Q/Q_max − 1, where Q_max = 120 NL/min
- m: injection position {0 = center, 1 = eccentric}

**Output** (4 channels): `(u, v, w, p)` in z-score normalized space

**Denormalization**: `u_phys = u_norm × σ_u + μ_u` (per-channel μ, σ stored in checkpoint)

## Performance

### center_8x384.pt (Center-blowing only)
| Q (NL/min) | Role | R² | u R² | v R² | w R² | p R² |
|------------|------|-----|------|------|------|------|
| 40 | Train | 0.975 | 0.94 | — | 0.99 | 1.00 |
| 60 | Train | 0.983 | 0.94 | — | 0.99 | 1.00 |
| 80 | Train | 0.987 | 0.94 | — | 0.99 | 1.00 |
| 100 | Extrap. | 0.988 | 0.94 | — | 0.99 | 1.00 |
| 120 | Extrap. | 0.984 | 0.94 | — | 0.99 | 1.00 |

### joint_8x384.pt (Center + Eccentric)
| Position | Q (NL/min) | R² |
|----------|------------|-----|
| Center | 40–120 | 0.974 |
| Eccentric | 40 | 0.940 |
| Eccentric | 80 | 0.955 |
| Eccentric | 120 | 0.969 |

## Training Details
- **Optimizer**: Adam, lr = 10⁻³, cosine annealing to 10⁻⁶
- **Batch size**: 8192
- **Epochs**: 2500 (center), 2000 (joint)
- **Loss**: MSE on velocity components (u, v, w)
- **Normalization**: z-score per output channel
- **Hardware**: NVIDIA RTX 5070 Laptop 8GB
- **Training time**: ~5 min (center), ~4 min (joint)

## Limitations
- Steady-state flow only (t = 300 s CFD solution)
- Single ladle geometry (100-ton class, truncated cone)
- Q ∈ [40, 120] NL/min range
- No explicit bubble phase prediction
- Pressure is auxiliary output (not primary target)

## Citation
Liu Y. A CFD-Informed Surrogate for Rapid Flow Prediction in an Argon-Stirred Ladle. *Advances in Manufacturing*, 2026 (under review).

## License
MIT
