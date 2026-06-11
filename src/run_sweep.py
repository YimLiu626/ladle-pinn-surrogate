"""
Sweep λ_phys × λ_bc on GPU. Phase 1 complete: best at lp=1.0, lb=1.0.
Phase 2: refine around best with more epochs.
"""
import sys, os, time, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from config import *
from data_loader import build_datasets
from model import PINN


def r2_mag(pred, true, norm):
    pd = pred.copy(); td = true.copy()
    for j, k in enumerate(['u_std','v_std','w_std']):
        pd[:,j] = pd[:,j]*norm[k]+norm[k.replace('std','mean')]
        td[:,j] = td[:,j]*norm[k]+norm[k.replace('std','mean')]
    mp = np.sqrt((pd[:,:3]**2).sum(1)); mt = np.sqrt((td[:,:3]**2).sum(1))
    return float(1.0 - ((mt-mp)**2).sum() / max(((mt-mt.mean())**2).sum(), 1e-20))


def per_field_r2(pred, true, norm):
    pd = pred.copy(); td = true.copy()
    out = {}
    for j, name in enumerate(['u','v','w','p']):
        k_std = f'{name}_std'; k_mean = f'{name}_mean'
        pd[:,j] = pd[:,j]*norm[k_std]+norm[k_mean]
        td[:,j] = td[:,j]*norm[k_std]+norm[k_mean]
        ssr = ((td[:,j]-pd[:,j])**2).sum()
        sst = ((td[:,j]-td[:,j].mean())**2).sum()
        out[name] = float(1.0 - ssr/max(sst,1e-20))
    return out


def train_and_eval(X_tr, Y_tr, Xev_i, Yev_i, Xev_e, Yev_e, wall, norm,
                   lambda_bc, epochs, phase1):
    """Single training run, returns R2s and per-field scores."""
    N = X_tr.shape[0]; DEV = X_tr.device
    model = PINN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, epochs+1):
        if ep <= phase1:
            lb = 0.0
        else:
            RAMP = 100
            ramp = min(1.0, (ep - phase1) / RAMP)
            lb = lambda_bc * ramp

        i1 = torch.randint(0, N, (BATCH_DATA,), device=DEV)
        # Plain MSE data loss
        total = ((model(X_tr[i1]) - Y_tr[i1])**2).mean()

        # Wall BC loss (no-slip: u=v=w=0)
        if lb > 0 and wall is not None:
            nw = min(wall.shape[0], BATCH_DATA)
            wb = wall[torch.randint(0, wall.shape[0], (nw,), device=DEV)]
            total = total + (model(wb)[:,:3].pow(2).sum(dim=-1)).mean() * lb

        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        r2i = r2_mag(model(Xev_i).cpu().numpy(), Yev_i.cpu().numpy(), norm)
        r2e = r2_mag(model(Xev_e).cpu().numpy(), Yev_e.cpu().numpy(), norm)
        pfi = per_field_r2(model(Xev_i).cpu().numpy(), Yev_i.cpu().numpy(), norm)
        pfe = per_field_r2(model(Xev_e).cpu().numpy(), Yev_e.cpu().numpy(), norm)
    return r2i, r2e, pfi, pfe


if __name__ == "__main__":
    print("Loading data...", flush=True)
    data = build_datasets()
    X_tr, Y_tr = data['X_train'], data['Y_train']
    X_int, Y_int = data['X_interp'], data['Y_interp']
    X_ext, Y_ext = data['X_extrap'], data['Y_extrap']
    wall = data['bc'].get('wall')
    norm = data['norm']
    DEV = X_tr.device
    print(f"Train: {X_tr.shape[0]:,}, Device: {DEV}, Wall: {wall.shape[0] if wall is not None else 0}")

    # Fixed eval sets
    torch.manual_seed(42)
    n_ev = min(50000, len(X_int))
    Xev_i, Yev_i = X_int[torch.randint(0, len(X_int), (n_ev,), device=DEV)], Y_int[torch.randint(0, len(X_int), (n_ev,), device=DEV)]
    n_ev_e = min(50000, len(X_ext))
    Xev_e, Yev_e = X_ext[torch.randint(0, len(X_ext), (n_ev_e,), device=DEV)], Y_ext[torch.randint(0, len(X_ext), (n_ev_e,), device=DEV)]

    all_results = []
    t_total = time.time()
    EPOCHS = 1200
    PHASE1_EP = 500

    # === BC weight sweep only (physics ablations separate) ===
    bc_vals = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    total_runs = len(bc_vals)

    print(f"\n{'='*60}")
    print(f"BC SWEEP: {total_runs} values, {EPOCHS}ep each (PHASE1={PHASE1_EP})")
    print(f"λ_bc ∈ {bc_vals}")
    print(f"{'='*60}")

    best_r2, best_lb = -99, 0.0

    for lb in bc_vals:
        t1 = time.time()
        print(f"  lb={lb:.4f} ...", end=" ", flush=True)
        r2i, r2e, pfi, pfe = train_and_eval(
            X_tr, Y_tr, Xev_i, Yev_i, Xev_e, Yev_e, wall, norm,
            lb, EPOCHS, PHASE1_EP)
        dt = time.time()-t1
        print(f"R2_int={r2i:.4f} R2_ext={r2e:.4f} | u={pfi['u']:.3f} w={pfi['w']:.3f} p={pfi['p']:.3f} | {dt:.0f}s")
        all_results.append({'lb':lb,'r2i':r2i,'r2e':r2e,'pfi':pfi,'pfe':pfe})
        if r2i > best_r2: best_r2, best_lb = r2i, lb

        # Save checkpoint every 3 runs
        if len(all_results) % 3 == 0:
            with open(os.path.join(RESULT_DIR,'sweep_bc.json'),'w') as f:
                json.dump({'results':all_results,'best':{'lb':best_lb,'r2':best_r2}}, f, indent=2)

    total_t = (time.time()-t_total)/60
    final = {'results':all_results,'best':{'lb':best_lb,'r2':best_r2},'time_min':total_t}
    with open(os.path.join(RESULT_DIR,'sweep_bc.json'),'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_results)} runs in {total_t:.1f} min")
    print(f"BEST: lb={best_lb}, R2_int={best_r2:.4f}")
    print(f"Saved: {os.path.join(RESULT_DIR,'sweep_all.json')}")
    print(f"{'='*60}")
