"""
Inference demo for pre-trained ladle flow surrogate models.
Loads the 8x384 MLP and predicts flow field for a given (x,y,z,Q,m).

Usage:
    python inference_demo.py --model center_8x384 --Q 80 --m 0
    python inference_demo.py --model joint_8x384 --Q 120 --m 1
"""
import sys, argparse, torch, numpy as np
sys.path.insert(0, '../../src')

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


def load_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = BigMLP()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    norm = ckpt['norm']
    return model, norm


def predict(model, norm, x, y, z, Q, m=0):
    """Predict (u, v, w, p) at a single point."""
    # Normalize input
    Lx = norm['Lx']; Ly = norm['Ly']; Lz = norm['Lz']
    x_n = (x - norm['x_min']) / Lx * 2 - 1
    y_n = (y - norm['y_min']) / Ly * 2 - 1
    z_n = (z - norm['z_min']) / Lz * 2 - 1
    Q_n = Q / norm['Q_max'] * 2 - 1

    X = torch.tensor([[x_n, y_n, z_n, Q_n, m]], dtype=torch.float32)
    with torch.no_grad():
        out = model(X).numpy()[0]

    # Denormalize
    u = out[0] * norm['u_std'] + norm['u_mean']
    v = out[1] * norm['v_std'] + norm['v_mean']
    w = out[2] * norm['w_std'] + norm['w_mean']
    p = out[3] * norm['p_std'] + norm['p_mean']
    return u, v, w, p


def predict_field(model, norm, csv_path, Q, m, output_path=None):
    """Predict full field from CFD input CSV."""
    import csv as csv_module
    xs, ys, zs = [], [], []
    with open(csv_path) as f:
        r = csv_module.DictReader(f, skipinitialspace=True)
        for row in r:
            xs.append(float(row['x-coordinate']))
            ys.append(float(row['y-coordinate']))
            zs.append(float(row['z-coordinate']))
    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    n = len(xs)

    Lx = norm['Lx']; Ly = norm['Ly']; Lz = norm['Lz']
    x_n = (xs - norm['x_min']) / Lx * 2 - 1
    y_n = (ys - norm['y_min']) / Ly * 2 - 1
    z_n = (zs - norm['z_min']) / Lz * 2 - 1
    Q_n = np.full(n, Q / norm['Q_max'] * 2 - 1)
    m_n = np.full(n, m)

    X = torch.tensor(np.column_stack([x_n, y_n, z_n, Q_n, m_n]), dtype=torch.float32)
    with torch.no_grad():
        out = model(X).numpy()

    u = out[:, 0] * norm['u_std'] + norm['u_mean']
    v = out[:, 1] * norm['v_std'] + norm['v_mean']
    w = out[:, 2] * norm['w_std'] + norm['w_mean']
    mag = np.sqrt(u**2 + v**2 + w**2)

    if output_path:
        np.savetxt(output_path, np.column_stack([xs, ys, zs, u, v, w, mag]),
                   header='x,y,z,u,v,w,|U|', delimiter=',', comments='')
        print(f'Prediction saved to {output_path}')

    return mag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ladle flow surrogate inference')
    parser.add_argument('--model', type=str, default='center_8x384',
                        choices=['center_8x384', 'joint_8x384', 'm0_8x384'],
                        help='Model to use')
    parser.add_argument('--Q', type=float, default=80, help='Gas flow rate (NL/min)')
    parser.add_argument('--m', type=int, default=0, help='Injection position (0=center, 1=eccentric)')
    parser.add_argument('--x', type=float, default=0.0, help='X coordinate (m)')
    parser.add_argument('--y', type=float, default=0.0, help='Y coordinate (m)')
    parser.add_argument('--z', type=float, default=1.0, help='Z coordinate (m)')
    parser.add_argument('--csv', type=str, default=None, help='CSV file for full field prediction')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    model_path = f'../models/{args.model}.pt'
    model, norm = load_model(model_path)
    print(f'Loaded: {model_path}')
    print(f'Model: 8x384 SiLU MLP, 1.04M parameters')

    # Single-point prediction
    u, v, w, p = predict(model, norm, args.x, args.y, args.z, args.Q, args.m)
    mag = np.sqrt(u**2 + v**2 + w**2)
    print(f'\nPoint prediction at (x={args.x}, y={args.y}, z={args.z}):')
    print(f'  Q = {args.Q} NL/min, m = {args.m}')
    print(f'  u = {u:.4f} m/s')
    print(f'  v = {v:.4f} m/s')
    print(f'  w = {w:.4f} m/s')
    print(f'  |U| = {mag:.4f} m/s')
    print(f'  p = {p:.1f} Pa')

    # Full field
    if args.csv:
        print(f'\nPredicting full field from {args.csv}...')
        mag = predict_field(model, norm, args.csv, args.Q, args.m, args.output)
        print(f'  Max |U| = {mag.max():.3f} m/s')
        print(f'  Mean |U| = {mag.mean():.4f} m/s')
