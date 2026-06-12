"""
Unified benchmark: evaluate all pre-trained models on standard test cases.
Outputs JSON with R-squared scores per model, Q, and velocity component.
"""
import sys, os, json, time, csv
sys.path.insert(0, '../../src')
import torch, numpy as np
from data_loader import normalize_x, normalize_y

MODEL_DIR = '../models'
DATA_DIR = '../../data/cfd/bottom'
SIDE_DIR = '../../data/cfd/side'


class BigMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [5] + [384] * 8 + [4]
        net = []
        for i in range(len(layers) - 1):
            net.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(torch.nn.SiLU())
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def load_cfd(path, Q, m):
    xs, ys, zs, us, vs, ws = [], [], [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f, skipinitialspace=True)
        for row in r:
            xs.append(float(row['x-coordinate']))
            ys.append(float(row['y-coordinate']))
            zs.append(float(row['z-coordinate']))
            us.append(float(row['x-velocity']))
            vs.append(float(row['y-velocity']))
            ws.append(float(row['z-velocity']))
    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    n = len(xs)
    mag_cfd = np.sqrt(np.array(us)**2 + np.array(vs)**2 + np.array(ws)**2)
    Xr = np.column_stack([xs, ys, zs, np.full(n, Q), np.full(n, m)])
    Yr = np.column_stack([us, vs, ws, np.zeros(n)])
    return Xr, Yr, mag_cfd


def r2_mag(pred, true):
    mp = np.sqrt((pred[:, :3]**2).sum(1))
    mt = true
    ssr = ((mt - mp)**2).sum()
    sst = ((mt - mt.mean())**2).sum()
    return float(1 - ssr / max(sst, 1e-20))


def evaluate(model_path, test_cases, DEV='cpu'):
    ckpt = torch.load(model_path, map_location=DEV, weights_only=False)
    model = BigMLP().to(DEV)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    norm = ckpt['norm']

    results = {}
    for name, path, Q, m in test_cases:
        Xr, Yr, mag_cfd = load_cfd(path, Q, m)
        Xt = torch.tensor(normalize_x(Xr, norm), dtype=torch.float32, device=DEV)
        with torch.no_grad():
            pred = model(Xt).cpu().numpy()
        for j, nm in enumerate(['u', 'v', 'w', 'p']):
            ks, km = f'{nm}_std', f'{nm}_mean'
            pred[:, j] = pred[:, j] * norm[ks] + norm[km]
        r2 = r2_mag(pred, mag_cfd)
        results[name] = r2
        print(f'  {name:25s}: R2 = {r2:.4f}')
    return results


if __name__ == '__main__':
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEV}\n')

    test_cases = [
        ('Center Q40',  f'{DATA_DIR}/q40-1.csv',  40, 0),
        ('Center Q60',  f'{DATA_DIR}/q60-1.csv',  60, 0),
        ('Center Q80',  f'{DATA_DIR}/q80-1.csv',  80, 0),
        ('Center Q100', f'{DATA_DIR}/q100-1.csv', 100, 0),
        ('Center Q120', f'{DATA_DIR}/q120-1.csv', 120, 0),
        ('Ecc. Q40',    f'{SIDE_DIR}/side-q40.csv',  40, 1),
        ('Ecc. Q80',    f'{SIDE_DIR}/side-q80.csv',  80, 1),
        ('Ecc. Q120',   f'{SIDE_DIR}/side-q120.csv', 120, 1),
    ]

    all_results = {}
    for model_name in ['center_8x384', 'joint_8x384']:
        path = os.path.join(MODEL_DIR, f'{model_name}.pt')
        if not os.path.exists(path):
            print(f'Skipping {model_name}: not found')
            continue
        print(f'=== {model_name} ===')
        t0 = time.time()
        r = evaluate(path, test_cases, DEV)
        all_results[model_name] = {'results': r, 'time_s': round(time.time() - t0, 1)}
        print(f'  Time: {all_results[model_name]["time_s"]}s\n')

    out_path = 'benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved: {out_path}')
