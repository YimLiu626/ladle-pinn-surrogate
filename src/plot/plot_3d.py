"""
3D visualizations for paper: Plume iso-surface + Steel phase velocity volume.
Generates both styles for all Q values.
"""
import csv, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'cfd', 'bottom')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def load_q(csv_path):
    xs, ys, zs, us, vs, ws, sts = [], [], [], [], [], [], []
    with open(csv_path) as f:
        r = csv.DictReader(f, skipinitialspace=True)
        for row in r:
            xs.append(float(row['x-coordinate']))
            ys.append(float(row['y-coordinate']))
            zs.append(float(row['z-coordinate']))
            us.append(float(row['x-velocity']))
            vs.append(float(row['y-velocity']))
            ws.append(float(row['z-velocity']))
            sts.append(float(row['steel-vof']))
    xs=np.array(xs); ys=np.array(ys); zs=np.array(zs)
    mag = np.sqrt(np.array(us)**2 + np.array(vs)**2 + np.array(ws)**2)
    return xs, ys, zs, mag, np.array(sts)


def plot_plume_iso(xs, ys, zs, mag, steel, q_label, out_path):
    """Plume core: steel region with |U| > 0.05 m/s in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mask_plume = (steel > 0.5) & (mag > 0.03)
    n_total = mask_plume.sum()
    if n_total < 1000:
        print(f'  Warning: only {n_total} plume cells')
        plt.close()
        return False

    n_plot = min(40000, n_total)
    idx = np.random.default_rng(42).choice(n_total, n_plot, replace=False)
    xp = xs[mask_plume][idx]; yp = ys[mask_plume][idx]
    zp = zs[mask_plume][idx]; cp = mag[mask_plume][idx]

    sc = ax.scatter(xp, yp, zp, c=cp, s=0.3, cmap='inferno',
                    alpha=0.7, vmax=0.15, rasterized=True)
    ax.view_init(elev=20, azim=-30)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_zlim(0, 1.85)
    ax.set_title(f'Q={q_label} NL/min | Plume core (|U|>0.03 m/s)')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label='|U| (m/s)', pad=0.08)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {os.path.basename(out_path)}')
    return True


def plot_steel_volume(xs, ys, zs, mag, steel, q_label, out_path):
    """Full steel phase 3D volume with velocity coloring."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mask_s = steel > 0.5
    n_show = min(100000, mask_s.sum())
    idx = np.random.default_rng(42).choice(mask_s.sum(), n_show, replace=False)
    x_s = xs[mask_s][idx]; y_s = ys[mask_s][idx]; z_s = zs[mask_s][idx]
    c_s = mag[mask_s][idx]

    sc = ax.scatter(x_s, y_s, z_s, c=c_s, s=0.2, cmap='jet',
                    alpha=0.4, vmax=0.15, rasterized=True)
    ax.view_init(elev=20, azim=-35)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Q={q_label} NL/min | Steel phase velocity field')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label='|U| (m/s)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {os.path.basename(out_path)}')


if __name__ == '__main__':
    for q in ['40', '60', '80', '100', '120']:
        path = os.path.join(DATA_DIR, f'q{q}-1.csv')
        if not os.path.exists(path):
            print(f'Skipping Q{q}')
            continue
        print(f'Q{q}...')
        xs, ys, zs, mag, steel = load_q(path)
        plot_plume_iso(xs, ys, zs, mag, steel, q,
                       os.path.join(OUT_DIR, f'3d_plume_Q{q}.png'))
        plot_steel_volume(xs, ys, zs, mag, steel, q,
                          os.path.join(OUT_DIR, f'3d_volume_Q{q}.png'))
    print('Done.')
