"""
Ablation studies: νt, physics terms, VOF strategy, spatial decay.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(8)
import numpy as np
from config import *
from data_loader import build_datasets
from model import PINN
from train_pinn import compute_losses, evaluate_r2


def train_ablation(data, lambda_phys, lambda_bc, nu_t_mode, epochs=1000,
                   vof_mode="soft", spatial_decay="cosine"):
    """Single ablation training."""
    X_tr, Y_tr = data['X_train'], data['Y_train']
    Wd_tr = data['wd_train'] if vof_mode == "soft" else torch.ones(data['n_train'])
    Wp_tr = data['wp_train'] if vof_mode == "soft" else torch.ones(data['n_train'])
    X_int, Y_int = data['X_interp'], data['Y_interp']
    X_ext, Y_ext = data['X_extrap'], data['Y_extrap']
    wall = data['bc'].get('wall')
    norm = data['norm']
    N = X_tr.shape[0]

    model = PINN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_r2, wait = -1.0, 0

    for ep in range(1, epochs + 1):
        lp = 0.0 if ep <= PHASE1 else lambda_phys
        lb = 0.0 if ep <= PHASE1 else lambda_bc

        i1 = torch.randint(0, N, (BATCH_DATA,))
        i2 = torch.randint(0, N, (BATCH_PHYS,))
        nw = min(wall.shape[0], BATCH_DATA) if wall is not None else 0
        wb = wall[torch.randint(0, wall.shape[0], (nw,))] if wall is not None else None

        total, _ = compute_losses(model, X_tr[i1], Y_tr[i1], Wd_tr[i1],
                                   X_tr[i2], Wp_tr[i2], wb, lp, lb, nu_t_mode)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 200 == 0 or ep == epochs:
            r2i, _ = evaluate_r2(model, X_int, Y_int, norm, batch_size=8192)
            if r2i > best_r2:
                best_r2 = r2i; wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                break

    r2i, pfi = evaluate_r2(model, X_int, Y_int, norm, batch_size=8192)
    r2e, pfe = evaluate_r2(model, X_ext, Y_ext, norm, batch_size=8192)
    return r2i, r2e, pfi, pfe


if __name__ == "__main__":
    print("Loading data...", flush=True)
    data = build_datasets()
    print("Ready.\n")

    results = {}
    t_total = time.time()

    # --- Exp 2: νt ablation ---
    print("=" * 50)
    print("Exp 2: nu_t ablation")
    print("=" * 50)
    for mode in ["mixing", "const", "none"]:
        t1 = time.time()
        print(f"  nu_t={mode} ...", end=" ", flush=True)
        r2i, r2e, pfi, pfe = train_ablation(data, LAMBDA_PHYS, LAMBDA_BC, mode, epochs=1000)
        dt = time.time() - t1
        print(f"R2_int={r2i:.4f} R2_ext={r2e:.4f} | u={pfi['u']:.3f} w={pfi['w']:.3f} | {dt:.0f}s")
        results[f"nut_{mode}"] = {'r2i': r2i, 'r2e': r2e, 'pfi': pfi, 'pfe': pfe}

    # --- Exp 3: Physics term ablation ---
    print("\n" + "=" * 50)
    print("Exp 3: Physics term ablation")
    print("=" * 50)
    configs = [
        ("data_only", 0, 0),
        ("data+BC", 0, LAMBDA_BC),
        ("data+BC+cont+mom", LAMBDA_PHYS, LAMBDA_BC),
    ]
    for name, lp, lb in configs:
        t1 = time.time()
        print(f"  {name} ...", end=" ", flush=True)
        r2i, r2e, pfi, pfe = train_ablation(data, lp, lb, "mixing", epochs=1000)
        dt = time.time() - t1
        print(f"R2_int={r2i:.4f} R2_ext={r2e:.4f} | {dt:.0f}s")
        results[f"phys_{name}"] = {'r2i': r2i, 'r2e': r2e, 'pfi': pfi, 'pfe': pfe}

    # --- Exp 8: VOF ablation ---
    print("\n" + "=" * 50)
    print("Exp 8: VOF ablation")
    print("=" * 50)
    for vof in ["soft", "hard", "none"]:
        t1 = time.time()
        print(f"  VOF={vof} ...", end=" ", flush=True)
        r2i, r2e, pfi, pfe = train_ablation(data, LAMBDA_PHYS, LAMBDA_BC, "mixing",
                                             epochs=1000, vof_mode=vof)
        dt = time.time() - t1
        print(f"R2_int={r2i:.4f} R2_ext={r2e:.4f} | {dt:.0f}s")
        results[f"vof_{vof}"] = {'r2i': r2i, 'r2e': r2e, 'pfi': pfi, 'pfe': pfe}

    total_t = (time.time() - t_total) / 60
    with open(os.path.join(RESULT_DIR, 'ablation.json'), 'w') as f:
        json.dump({'results': results, 'total_time_min': total_t}, f, indent=2)
    print(f"\nAll ablations done: {len(results)} configs in {total_t:.1f} min")
