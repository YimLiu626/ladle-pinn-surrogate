# PINN Model Zoo — Ladle Flow Surrogate

Pre-trained surrogate models for rapid 3D flow prediction in argon-stirred ladle furnaces.

## Available Models

| Model | Architecture | Training Data | R² Center | R² Ecc. | Download |
|-------|-------------|---------------|-----------|----------|----------|
| `center_8x384` | 8×384 SiLU MLP | Center Q40,60,80 | 0.983 | — | [pt](models/center_8x384.pt) |
| `joint_8x384` | 8×384 SiLU MLP | Center + Ecc. | 0.974 | 0.94–0.97 | [pt](models/joint_8x384.pt) |
| `m0_8x384` | 8×384 SiLU MLP | Center (m=0 only) | 0.980 | — | [pt](models/m0_8x384.pt) |

Full model details → [MODEL_CARD.md](models/MODEL_CARD.md)

## Quick Inference

```python
import torch
import sys; sys.path.insert(0, '../src')
from model_zoo.demo.inference_demo import load_model, predict

model, norm = load_model('models/center_8x384.pt')

# Predict velocity at a single point
u, v, w, p = predict(model, norm, x=0.0, y=0.0, z=1.0, Q=80, m=0)
print(f'|U| = {(u**2 + v**2 + w**2)**0.5:.3f} m/s')
```

Or from command line:

```bash
python demo/inference_demo.py --model center_8x384 --Q 80 --m 0 --x 0 --y 0 --z 1.0
```

## Benchmark

```bash
cd benchmark
python benchmark.py
```

Evaluates all models on 8 test cases (5 center + 3 eccentric). Outputs `benchmark_results.json`.

## Input/Output

| | Dimension | Description |
|---|-----------|-------------|
| Input | (x, y, z, Q, m) | Spatial coords [−1,1], flow rate [−1,1], position ∈ {0,1} |
| Output | (u, v, w, p) | Velocity components + pressure (z-score normalized) |

## Requirements

```
torch >= 2.0
numpy
```

## License

MIT — use, modify, and distribute freely.

## Related Repositories

- [ladle-pinn-surrogate](https://github.com/YimLiu626/ladle-pinn-surrogate) — Full training code and experiments
- [ladle-cfd-bucket-screening](https://github.com/YimLiu626/-ladle-cfd-bucket-screening) — CFD-DPM data generation
