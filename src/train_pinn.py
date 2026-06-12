"""
train_pinn.py — PINN 训练主脚本
Two-phase: Phase 1 纯数据 → Phase 2 数据+物理+BC
VOF 软加权 + 空间衰减 + 混合长度 ν_t + wall BC

Usage: python src/train_pinn.py [--sweep] [--ablation MODE]
"""
import os, sys, time, json, argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from data_loader import build_datasets, normalize_x, normalize_y
from model import PINN
from physics import compute_rans_residuals


def compute_losses(model, X_data, Y_data, Wd, X_phys, Wp, wall_tensor,
                   lambda_phys, lambda_bc, nu_t_mode="mixing"):
    """计算加权总损失。返回 total, L_data, L_phys, L_bc。"""
    losses = {}

    # Data loss — VOF 软加权
    if X_data is not None and X_data.shape[0] > 0:
        pred = model(X_data)
        err = (pred - Y_data).pow(2).sum(dim=-1)  # (B,)
        L_data = (err * Wd).mean()
    else:
        L_data = torch.tensor(0.0, device=DEVICE)
    losses['data'] = L_data

    # Physics loss
    if lambda_phys > 0 and X_phys is not None and X_phys.shape[0] > 0:
        L_phys_raw, _, _, _, _ = compute_rans_residuals(model, X_phys, nu_t_mode)
        L_phys = L_phys_raw * lambda_phys
    else:
        L_phys = torch.tensor(0.0, device=DEVICE)
    losses['phys'] = L_phys

    # BC loss — wall no-slip
    if lambda_bc > 0 and wall_tensor is not None and wall_tensor.shape[0] > 0:
        pred_wall = model(wall_tensor)
        # no-slip: u=v=w=0, p unconstrained
        L_bc = (pred_wall[:, :3].pow(2).sum(dim=-1)).mean() * lambda_bc
    else:
        L_bc = torch.tensor(0.0, device=DEVICE)
    losses['bc'] = L_bc

    total = L_data + L_phys + L_bc
    return total, losses


def evaluate_r2(model, X_test, Y_test, norm, batch_size=16384):
    """返回 overall R2 和 per-field R2（分 batch 避免 CPU OOM）。"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size]
            yb = Y_test[i:i+batch_size]
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    # Denormalize (z-score)
    pred_phys = pred.copy()
    true_phys = true.copy()
    pred_phys[:, 0] = pred_phys[:, 0] * norm['u_std'] + norm['u_mean']
    pred_phys[:, 1] = pred_phys[:, 1] * norm['v_std'] + norm['v_mean']
    pred_phys[:, 2] = pred_phys[:, 2] * norm['w_std'] + norm['w_mean']
    pred_phys[:, 3] = pred_phys[:, 3] * norm['p_std'] + norm['p_mean']
    true_phys[:, 0] = true_phys[:, 0] * norm['u_std'] + norm['u_mean']
    true_phys[:, 1] = true_phys[:, 1] * norm['v_std'] + norm['v_mean']
    true_phys[:, 2] = true_phys[:, 2] * norm['w_std'] + norm['w_mean']
    true_phys[:, 3] = true_phys[:, 3] * norm['p_std'] + norm['p_mean']

    mag_pred = np.sqrt((pred_phys[:, :3]**2).sum(axis=1))
    mag_true = np.sqrt((true_phys[:, :3]**2).sum(axis=1))

    ss_res = ((mag_true - mag_pred)**2).sum()
    ss_tot = ((mag_true - mag_true.mean())**2).sum()
    r2_mag = 1 - ss_res / max(ss_tot, 1e-20)

    per_field = {}
    for j, name in enumerate(['u', 'v', 'w', 'p']):
        ss_r = ((true_phys[:, j] - pred_phys[:, j])**2).sum()
        ss_t = ((true_phys[:, j] - true_phys[:, j].mean())**2).sum()
        per_field[name] = 1 - ss_r / max(ss_t, 1e-20)

    model.train()
    return r2_mag, per_field


def train_one_config(lambda_phys=LAMBDA_PHYS, lambda_bc=LAMBDA_BC,
                     nu_t_mode="mixing", save_name=None, verbose=True):
    """单次训练，返回 best_R2 和 per-field 分数。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: λ_phys={lambda_phys}, λ_bc={lambda_bc}, νt={nu_t_mode}")
        print(f"{'='*60}")

    data = build_datasets()
    norm = data['norm']

    X_tr, Y_tr, Wd_tr, Wp_tr = data['X_train'], data['Y_train'], data['wd_train'], data['wp_train']
    X_int, Y_int = data['X_interp'], data['Y_interp']
    X_ext, Y_ext = data['X_extrap'], data['Y_extrap']
    wall = data['bc'].get('wall')

    # Phase 1 physics off
    lambda_phys_eff = 0.0
    lambda_bc_eff = 0.0
    current_phase = 1

    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_r2 = -1.0
    best_epoch = 0
    best_state = None
    best_r2_ext = -1.0
    wait = 0
    loss_history = []

    N_data = X_tr.shape[0]
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # Phase transition
        if epoch == PHASE1 + 1:
            lambda_phys_eff = lambda_phys
            lambda_bc_eff = lambda_bc
            current_phase = 2
            if verbose:
                print(f"  [Phase 2] Physics enabled: λ_phys={lambda_phys_eff}, λ_bc={lambda_bc_eff}")

        # Sample batches
        idx_data = torch.randint(0, N_data, (BATCH_DATA,), device=DEVICE)
        idx_phys = torch.randint(0, N_data, (BATCH_PHYS,), device=DEVICE)

        X_batch_data = X_tr[idx_data]
        Y_batch_data = Y_tr[idx_data]
        Wd_batch = Wd_tr[idx_data]
        X_batch_phys = X_tr[idx_phys]
        Wp_batch = Wp_tr[idx_phys]

        # Sample wall BC
        wall_batch = None
        if wall is not None and wall.shape[0] > 0:
            n_wall = min(wall.shape[0], BATCH_DATA)
            idx_wall = torch.randint(0, wall.shape[0], (n_wall,), device=DEVICE)
            wall_batch = wall[idx_wall]

        total, losses = compute_losses(
            model, X_batch_data, Y_batch_data, Wd_batch,
            X_batch_phys, wall_batch,
            lambda_phys_eff, lambda_bc_eff, nu_t_mode
        )

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % LOG_INTERVAL == 0:
            r2_int, pf_int = evaluate_r2(model, X_int, Y_int, norm)
            r2_ext, pf_ext = evaluate_r2(model, X_ext, Y_ext, norm)
            elapsed = time.time() - t0

            Ld = losses['data'].item()
            Lp = losses['phys'].item()
            Lb = losses['bc'].item()

            if verbose:
                print(f"  ep={epoch:5d} | "
                      f"L_data={Ld:.4e} L_phys={Lp:.4e} L_bc={Lb:.4e} | "
                      f"R2_int={r2_int:.4f} R2_ext={r2_ext:.4f} | "
                      f"u={pf_int['u']:.3f} w={pf_int['w']:.3f} p={pf_int['p']:.3f} | "
                      f"{elapsed:.0f}s")

            loss_history.append({
                'epoch': epoch, 'phase': current_phase,
                'L_data': Ld, 'L_phys': Lp, 'L_bc': Lb,
                'R2_interp': r2_int, 'R2_extrap': r2_ext,
                'per_field_int': pf_int, 'per_field_ext': pf_ext
            })

            # 用内插 R2 做 early stopping
            if r2_int > best_r2:
                best_r2 = r2_int
                best_r2_ext = r2_ext
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {epoch}, best R2_int={best_r2:.4f} (ep {best_epoch})")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    r2_int_final, pf_int = evaluate_r2(model, X_int, Y_int, norm)
    r2_ext_final, pf_ext = evaluate_r2(model, X_ext, Y_ext, norm)

    result = {
        'lambda_phys': lambda_phys, 'lambda_bc': lambda_bc,
        'nu_t_mode': nu_t_mode,
        'R2_interp': r2_int_final, 'R2_extrap': r2_ext_final,
        'per_field_interp': pf_int, 'per_field_extrap': pf_ext,
        'best_epoch': best_epoch, 'total_epochs': epoch,
        'n_train': N_data, 'Q_list': data['Q_list'],
        'norm': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in norm.items()},
        'n_params': sum(p.numel() for p in model.parameters()),
        'loss_history': loss_history,
    }

    if verbose:
        print(f"\n  Final: R2_int={r2_int_final:.4f} R2_ext={r2_ext_final:.4f}")
        print(f"  Per-field: u={pf_int['u']:.4f} v={pf_int['v']:.4f} w={pf_int['w']:.4f} p={pf_int['p']:.4f}")

    # Save model
    if save_name:
        os.makedirs(MODEL_DIR, exist_ok=True)
        save_path = os.path.join(MODEL_DIR, save_name)
        torch.save({
            'model_state': best_state if best_state else model.state_dict(),
            'config': result,
        }, save_path)
        if verbose:
            print(f"  Model saved: {save_path}")

    return result


def run_sweep():
    """Experiment 1: λ_phys × λ_bc grid sweep (63 combos)."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: λ_phys × λ_bc weight sweep")
    print("="*70)

    results = []
    for lp in SWEEP_PHYS:
        for lb in SWEEP_BC:
            r = train_one_config(
                lambda_phys=lp, lambda_bc=lb,
                nu_t_mode="mixing",
                verbose=False
            )
            results.append(r)
            print(f"  λ_phys={lp:5.2f} λ_bc={lb:5.2f} → R2_int={r['R2_interp']:.4f} R2_ext={r['R2_extrap']:.4f}")

    # Save results
    sweep_path = os.path.join(RESULT_DIR, 'sweep_lambda.json')
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep saved: {sweep_path}")

    # Find best
    best = max(results, key=lambda r: r['R2_interp'])
    print(f"Best: λ_phys={best['lambda_phys']}, λ_bc={best['lambda_bc']}, "
          f"R2_int={best['R2_interp']:.4f}, R2_ext={best['R2_extrap']:.4f}")
    return best


def run_ablation():
    """Experiments 2,3,8,11: ablation studies."""
    print("\n" + "="*70)
    print("EXPERIMENTS 2+3: νt + physics term ablation")
    print("="*70)

    results = {}

    # Exp 2: νt ablation
    for mode in ["mixing", "const", "none"]:
        r = train_one_config(nu_t_mode=mode, save_name=f"ablation_nut_{mode}.pt")
        results[f"nut_{mode}"] = r

    # Exp 3: Physics term ablation (data-only is MLP baseline)
    # data+BC
    r = train_one_config(lambda_phys=0, lambda_bc=1.0, nu_t_mode="mixing",
                         save_name="ablation_data_bc.pt")
    results["data_bc"] = r
    # data+BC+continuity
    # (implemented in full ablation script)

    ablation_path = os.path.join(RESULT_DIR, 'ablation.json')
    with open(ablation_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved: {ablation_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PINN for ladle flow')
    parser.add_argument('--sweep', action='store_true', help='Run λ sweep')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--lambda_phys', type=float, default=LAMBDA_PHYS)
    parser.add_argument('--lambda_bc', type=float, default=LAMBDA_BC)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    elif args.ablation:
        run_ablation()
    else:
        # Single training run
        save_name = args.save or f"pinn_lp{args.lambda_phys}_lb{args.lambda_bc}.pt"
        train_one_config(
            lambda_phys=args.lambda_phys,
            lambda_bc=args.lambda_bc,
            nu_t_mode="mixing",
            save_name=save_name
        )
