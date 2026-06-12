"""
Full experiments: νt ablation, multi-seed stability, per-Q predictions.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from config import *
from data_loader import build_datasets
from model import PINN

def r2_mag(pred, true, norm):
    pd, td = pred.copy(), true.copy()
    for j,k in enumerate(['u_std','v_std','w_std']):
        m=k.replace('std','mean')
        pd[:,j]=pd[:,j]*norm[k]+norm[m]
        td[:,j]=td[:,j]*norm[k]+norm[m]
    mp=np.sqrt((pd[:,:3]**2).sum(1)); mt=np.sqrt((td[:,:3]**2).sum(1))
    return float(1-((mt-mp)**2).sum()/max(((mt-mt.mean())**2).sum(),1e-20))

def per_field(pred, true, norm):
    pd,td=pred.copy(),true.copy()
    out={}
    for j,nm in enumerate(['u','v','w','p']):
        ks,km=f'{nm}_std',f'{nm}_mean'
        pd[:,j]=pd[:,j]*norm[ks]+norm[km]
        td[:,j]=td[:,j]*norm[ks]+norm[km]
        ssr=((td[:,j]-pd[:,j])**2).sum(); sst=((td[:,j]-td[:,j].mean())**2).sum()
        out[nm]=float(1-ssr/max(sst,1e-20))
    return out

def train_model(data, epochs=2000, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    N=len(X_tr); DEV=X_tr.device
    model=PINN().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(1,epochs+1):
        idx=torch.randint(0,N,(8192,),device=DEV)
        loss=((model(X_tr[idx])-Y_tr[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_model(model, data):
    norm=data['norm']
    with torch.no_grad():
        ri=r2_mag(model(data['X_interp']).cpu().numpy(),data['Y_interp'].cpu().numpy(),norm)
        re=r2_mag(model(data['X_extrap']).cpu().numpy(),data['Y_extrap'].cpu().numpy(),norm)
        pi=per_field(model(data['X_interp']).cpu().numpy(),data['Y_interp'].cpu().numpy(),norm)
        pe=per_field(model(data['X_extrap']).cpu().numpy(),data['Y_extrap'].cpu().numpy(),norm)
    return ri,re,pi,pe

def predict_all(data, model, out_dir):
    """Save predictions for all Q values."""
    os.makedirs(out_dir, exist_ok=True)
    # Group by Q
    all_Q = data['Q_list']
    X_all = torch.cat([data['X_interp'], data['X_extrap']])
    Y_all = torch.cat([data['Y_interp'], data['Y_extrap']])
    Q_vals = X_all[:,3]  # Q is column 3 in normalized space

    norm = data['norm']
    Qmax = norm['Q_max']

    with torch.no_grad():
        pred = model(X_all).cpu().numpy()
        true = Y_all.cpu().numpy()

    # Denormalize Q for grouping
    Q_phys = ((Q_vals.cpu().numpy() + 1) / 2 * Qmax).round()

    # Denormalize predictions
    pd, td = pred.copy(), true.copy()
    for j,nm in enumerate(['u','v','w','p']):
        ks,km=f'{nm}_std',f'{nm}_mean'
        pd[:,j]=pd[:,j]*norm[ks]+norm[km]
        td[:,j]=td[:,j]*norm[ks]+norm[km]

    for q in sorted(set(Q_phys)):
        mask = Q_phys == q
        n = mask.sum()
        # Get original coordinates (need to reload or use stored)
        print(f"  Q={int(q)}: {n:,} cells, R2_mag={r2_mag(pd[mask],td[mask],norm):.4f}")

if __name__=="__main__":
    print("Loading data...",flush=True)
    data=build_datasets()

    # 1. Multi-seed stability
    print("\n=== Multi-seed (3 seeds) ===")
    seeds=[42,123,456]
    seed_results=[]
    for s in seeds:
        t0=time.time()
        print(f"  seed={s}...",end=" ",flush=True)
        m=train_model(data,epochs=1500,seed=s)
        ri,re,pi,pe=eval_model(m,data)
        print(f"R2_int={ri:.4f} R2_ext={re:.4f} | u={pi['u']:.3f} w={pi['w']:.3f} | {time.time()-t0:.0f}s")
        seed_results.append({'seed':s,'r2i':ri,'r2e':re,'pfi':pi,'pfe':pe})

    ris=[r['r2i'] for r in seed_results]
    print(f"  Mean R2_int={np.mean(ris):.4f} +- {np.std(ris):.4f}")

    # 2. Save best model (seed=42, 2000ep)
    print("\n=== Best model (seed=42, 2000ep) ===")
    best_model=train_model(data,epochs=2000,seed=42)
    ri,re,pi,pe=eval_model(best_model,data)
    print(f"  R2_int={ri:.4f} R2_ext={re:.4f} | u={pi['u']:.3f} w={pi['w']:.3f} p={pi['p']:.3f}")
    torch.save({
        'model_state':best_model.state_dict(),
        'norm':data['norm'],
        'r2_int':ri,'r2_ext':re,
        'per_field_int':pi,'per_field_ext':pe,
        'seed_results':seed_results
    }, os.path.join(MODEL_DIR,'best_model.pt'))

    # 3. Per-Q predictions
    print("\n=== Per-Q predictions ===")
    predict_all(data, best_model, os.path.join(RESULT_DIR,'predictions'))

    # 4. Save all results
    all_res={
        'best':{'r2i':ri,'r2e':re,'pfi':pi,'pfe':pe},
        'multi_seed':seed_results,
    }
    with open(os.path.join(RESULT_DIR,'experiments.json'),'w') as f:
        json.dump(all_res,f,indent=2)
    print(f"\nSaved: {os.path.join(RESULT_DIR,'experiments.json')}")
