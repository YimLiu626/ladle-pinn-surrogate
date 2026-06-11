"""CFD field visualization: velocity contours and VOF distribution.
Usage: python plot_field.py <csv_path> [--q Q_value]
Output: velocity_contour_Q{}.png + vof_distribution_Q{}.png
"""
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_data(csv_path):
    data = {'x': [], 'y': [], 'z': [],
            'u': [], 'v': [], 'w': [], 'p': [],
            'steel': [], 'slag': [], 'argon': []}
    with open(csv_path) as f:
        r = csv.DictReader(f, skipinitialspace=True)
        for row in r:
            data['x'].append(float(row['x-coordinate']))
            data['y'].append(float(row['y-coordinate']))
            data['z'].append(float(row['z-coordinate']))
            data['u'].append(float(row['x-velocity']))
            data['v'].append(float(row['y-velocity']))
            data['w'].append(float(row['z-velocity']))
            data['p'].append(float(row['pressure']))
            data['steel'].append(float(row['steel-vof']))
            data['slag'].append(float(row['slag-vof']))
            data['argon'].append(float(row['argon-vof']))
    for k in data:
        data[k] = np.array(data[k])
    data['mag'] = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
    return data


def plot_velocity_contours(data, q_label, out_dir='.'):
    """YZ and XZ center-plane velocity magnitude + steel VOF overlay."""
    mask_yz = np.abs(data['x']) < 0.02
    mask_xz = np.abs(data['y']) < 0.02

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # YZ: velocity magnitude
    sc = axes[0, 0].scatter(data['y'][mask_yz], data['z'][mask_yz],
                            c=data['mag'][mask_yz], s=0.5, cmap='jet', vmax=0.15)
    axes[0, 0].axhline(y=1.85, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0, 0].set_xlabel('Y (m)')
    axes[0, 0].set_ylabel('Z (m)')
    axes[0, 0].set_title(f'Q={q_label} | YZ center plane: |U| (m/s)')
    plt.colorbar(sc, ax=axes[0, 0])

    # YZ: steel VOF
    sc1 = axes[0, 1].scatter(data['y'][mask_yz], data['z'][mask_yz],
                             c=data['steel'][mask_yz], s=0.5, cmap='Blues_r', vmin=0, vmax=1)
    axes[0, 1].axhline(y=1.85, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel('Y (m)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].set_title(f'Q={q_label} | YZ center plane: steel VOF')
    plt.colorbar(sc1, ax=axes[0, 1])

    # XZ: velocity magnitude
    sc2 = axes[1, 0].scatter(data['x'][mask_xz], data['z'][mask_xz],
                             c=data['mag'][mask_xz], s=0.5, cmap='jet', vmax=0.15)
    axes[1, 0].axhline(y=1.85, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_title(f'Q={q_label} | XZ center plane: |U| (m/s)')
    plt.colorbar(sc2, ax=axes[1, 0])

    # XZ: steel VOF
    sc3 = axes[1, 1].scatter(data['x'][mask_xz], data['z'][mask_xz],
                             c=data['steel'][mask_xz], s=0.5, cmap='Blues_r', vmin=0, vmax=1)
    axes[1, 1].axhline(y=1.85, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Z (m)')
    axes[1, 1].set_title(f'Q={q_label} | XZ center plane: steel VOF')
    plt.colorbar(sc3, ax=axes[1, 1])

    plt.tight_layout()
    out = f'{out_dir}/velocity_contour_Q{q_label}.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    return out


def plot_vof_distribution(data, q_label, out_dir='.'):
    """Vertical profile of phase fractions and velocity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    z_bins = np.arange(0, 2.6, 0.05)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    profiles = {'steel': [], 'slag': [], 'argon': [], 'mag': []}

    for zl, zh in zip(z_bins[:-1], z_bins[1:]):
        m = (data['z'] >= zl) & (data['z'] < zh)
        for key in profiles:
            profiles[key].append(data[key][m].mean() if m.sum() > 0 else 0)

    # Phase fractions
    axes[0].plot(profiles['steel'], z_centers, 'b-', linewidth=2, label='steel')
    axes[0].plot(profiles['slag'], z_centers, 'orange', linewidth=2, label='slag')
    axes[0].plot(profiles['argon'], z_centers, 'gray', linewidth=2, label='argon')
    axes[0].axhline(y=1.85, color='k', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Mean VOF')
    axes[0].set_ylabel('Z (m)')
    axes[0].set_title(f'Q={q_label} | Phase distribution')
    axes[0].legend()

    # Velocity profile
    axes[1].plot(profiles['mag'], z_centers, 'r-', linewidth=2)
    axes[1].axhline(y=1.85, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Mean |U| (m/s)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title(f'Q={q_label} | Velocity profile')

    plt.tight_layout()
    out = f'{out_dir}/vof_distribution_Q{q_label}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    return out


def print_summary(data):
    """Print vertical layer summary."""
    print(f'{"z~":>6s}  {"n":>6s}  {"steel":>8s}  {"slag":>8s}  {"argon":>8s}  {"|U|_mean":>10s}')
    print('-' * 56)
    for zr in np.arange(0, 2.6, 0.2):
        mz = np.abs(data['z'] - zr) < 0.1
        if mz.sum() > 0:
            print(f'{zr:5.1f}  {mz.sum():6d}  '
                  f'{data["steel"][mz].mean():8.3f}  {data["slag"][mz].mean():8.3f}  '
                  f'{data["argon"][mz].mean():8.3f}  {data["mag"][mz].mean():10.4f}')


def main():
    parser = argparse.ArgumentParser(description='CFD field visualization')
    parser.add_argument('csv_path', help='Path to case CSV file')
    parser.add_argument('--q', default='?', help='Q value label')
    parser.add_argument('--out', default='.', help='Output directory')
    args = parser.parse_args()

    data = load_data(args.csv_path)
    print_summary(data)
    plot_velocity_contours(data, args.q, args.out)
    plot_vof_distribution(data, args.q, args.out)


if __name__ == '__main__':
    main()
