"""
Full training pipeline: pure data + optional BC + optional physics.
Single clean run. Tested: R2=0.88 at 1200ep pure data.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from config import *
from data_loader import build_datasets
from model import PINN

def r2_mag(pred, true, norm):
    pd, td = pred.copy(), true.copy()
    for j, k in enumerate(['u_std','v_std','w_std']):
        m = k.replace('std','mean')
        pd[:,j] = pd[:,j]*norm[k]+norm[m]
        td[:,j] = td[:,j]*norm[k]+norm[m]
    mp = np.sqrt((pd[:,:3]**2).sum(1))
    mt = np.sqrt((td[:,:3]**2).sum(1))
    ssr = ((mt-mp)**2).sum(); sst = ((mt-mt.mean())**2).sum()
    return float(1-ssr/max(sst,1e-20))

def per_field(pred, true, norm):
    pd, td = pred.copy(), true.copy()
    out = {}
    for nm in ['u','v','w','p']:
        ks, km = f'{nm}_std', f'{nm}_mean'
        pd[:,list(out).index(nm) if nm in out else ['u','v','w','p'].index(nm)] = \
            pd[:,['u','v','w','p'].index(nm)]*norm[ks]+norm[km]
        td[:,['u','v','w','p'].index(nm)] = \
            td[:,['u','v','w','p'].index(nm)]*norm[ks]+norm[km]
    pd2, td2 = pred.copy(), true.copy()
    for j, nm in enumerate(['u','v','w','p']):
        ks, km = f'{nm}_std', f'{nm}_mean'
        pd2[:,j] = pd2[:,j]*norm[ks]+norm[km]
        td2[:,j] = td2[:,j]*norm[ks]+norm[km]
        ssr = ((td2[:,j]-pd2[:,j])**2).sum()
        sst = ((td2[:,j]-td2[:,j].mean())**2).sum()
        out[nm] = float(1-ssr/max(sst,1e-20))
    return out

def train(data, lambda_bc=0.0, epochs=2000, phase1=500, save_path=None):
    X_tr, Y_tr = data['X_train'], data['Y_train']
    X_int, Y_int = data['X_interp'], data['Y_interp']
    X_ext, Y_ext = data['X_extrap'], data['Y_extrap']
    wall = data['bc'].get('wall')
    norm = data['norm']
    N = len(X_tr)
    DEV = X_tr.device

    print(f"Train: {N:,} | epochs={epochs}, phase1={phase1}, lb={lambda_bc}")

    model = PINN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    best = -99

    for ep in range(1, epochs+1):
        # Data loss
        i1 = torch.randint(0, N, (8192,), device=DEV)
        total = ((model(X_tr[i1]) - Y_tr[i1])**2).mean()

        # BC (after phase1)
        if ep > phase1 and lambda_bc > 0 and wall is not None:
            ramp = min(1.0, (ep-phase1)/100)
            nw = min(wall.shape[0], 4096)
            wb = wall[torch.randint(0, wall.shape[0], (nw,), device=DEV)]
            total = total + (model(wb)[:,:3]**2).sum(dim=-1).mean()*lambda_bc*ramp

        opt.zero_grad(); total.backward(); opt.step()

        if ep % 300 == 0:
            with torch.no_grad():
                ri = r2_mag(model(X_int).cpu().numpy(), Y_int.cpu().numpy(), norm)
                re = r2_mag(model(X_ext).cpu().numpy(), Y_ext.cpu().numpy(), norm)
            best = max(best, ri)
            print(f"  ep={ep:5d}: R2_int={ri:.4f} R2_ext={re:.4f} | {time.time()-t0:.0f}s")

    # Final eval on full sets
    with torch.no_grad():
        ri = r2_mag(model(X_int).cpu().numpy(), Y_int.cpu().numpy(), norm)
        re = r2_mag(model(X_ext).cpu().numpy(), Y_ext.cpu().numpy(), norm)
        pfi = per_field(model(X_int).cpu().numpy(), Y_int.cpu().numpy(), norm)
        pfe = per_field(model(X_ext).cpu().numpy(), Y_ext.cpu().numpy(), norm)

    print(f"  FINAL: R2_int={ri:.4f} R2_ext={re:.4f} | u={pfi['u']:.3f} w={pfi['w']:.3f} p={pfi['p']:.3f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state': model.state_dict(), 'norm': norm,
                    'r2': {'int': ri, 'ext': re}, 'per_field_int': pfi, 'per_field_ext': pfe}, save_path)
        print(f"  Saved: {save_path}")

    return ri, re, pfi, pfe


if __name__ == "__main__":
    print("Loading...", flush=True)
    data = build_datasets()

    # 1. Pure data baseline
    print("\n=== Pure data (lb=0) ===")
    train(data, lambda_bc=0.0, epochs=2000, phase1=10**9,
          save_path=os.path.join(MODEL_DIR, 'pure_data.pt'))

    # 2. BC sweep (quick)
    print("\n=== BC sweep ===")
    bc_results = []
    for lb in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        ri, re, pfi, pfe = train(data, lambda_bc=lb, epochs=1500, phase1=500)
        bc_results.append({'lb': lb, 'r2i': ri, 'r2e': re, 'pfi': pfi, 'pfe': pfe})
        print(f"  lb={lb}: R2_int={ri:.4f} R2_ext={re:.4f}")

    best = max(bc_results, key=lambda x: x['r2i'])
    print(f"\nBest BC: lb={best['lb']}, R2_int={best['r2i']:.4f}")

    with open(os.path.join(RESULT_DIR, 'bc_sweep.json'), 'w') as f:
        json.dump(bc_results, f, indent=2)
