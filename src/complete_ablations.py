"""
Complete ablations with 6x128 model (fast) and continuity-only physics.
Trends are what matter — large model used for final metrics.
"""
import sys,os,time,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,numpy as np
from config import *
from data_loader import build_datasets
from model import PINN

def r2_mag(t,p,norm):
    td,pd=t.copy(),p.copy()
    for j,k in enumerate(['u_std','v_std','w_std']):
        m=k.replace('std','mean')
        pd[:,j]=pd[:,j]*norm[k]+norm[m];td[:,j]=td[:,j]*norm[k]+norm[m]
    mp=np.sqrt((pd[:,:3]**2).sum(1));mt=np.sqrt((td[:,:3]**2).sum(1))
    return float(1-((mt-mp)**2).sum()/max(((mt-mt.mean())**2).sum(),1e-20))

def per_field(p,t,n):
    pd,td=p.copy(),t.copy()
    out={}
    for j,nm in enumerate(['u','v','w','p']):
        ks,km=f'{nm}_std',f'{nm}_mean'
        pd[:,j]=pd[:,j]*n[ks]+n[km];td[:,j]=td[:,j]*n[ks]+n[km]
        ssr=((td[:,j]-pd[:,j])**2).sum();sst=((td[:,j]-td[:,j].mean())**2).sum()
        out[nm]=float(1-ssr/max(sst,1e-20))
    return out

def continuity_loss(model, X, norm):
    """Physical-space continuity loss. Fast: 3 grad calls."""
    X=X.detach().clone().requires_grad_(True)
    out=model(X)
    gu=torch.autograd.grad(out[:,0].sum(),X,create_graph=True)[0]
    gv=torch.autograd.grad(out[:,1].sum(),X,create_graph=True)[0]
    gw=torch.autograd.grad(out[:,2].sum(),X,create_graph=True)[0]
    Lx,Ly,Lz=norm['Lx'],norm['Ly'],norm['Lz']
    du_dx=gu[:,0]*2*norm['u_std']/Lx
    dv_dy=gv[:,1]*2*norm['v_std']/Ly
    dw_dz=gw[:,2]*2*norm['w_std']/Lz
    cont=du_dx+dv_dy+dw_dz
    return cont.pow(2).mean()

def train_abl(data, lb=0, lp=0, nu_mode="mixing", vof_mode="filtered",
              epochs=1200, phase1=500, seed=42):
    torch.manual_seed(seed);np.random.seed(seed)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    X_int,Y_int=data['X_interp'],data['Y_interp']
    X_ext,Y_ext=data['X_extrap'],data['Y_extrap']
    wall=data['bc'].get('wall')
    norm=data['norm']
    N=len(X_tr);DEV=X_tr.device

    if vof_mode=="none": Wd=torch.ones(N,device=DEV)
    elif vof_mode=="hard": Wd=(data['wd_train']>0).float()
    else: Wd=torch.ones(N,device=DEV)

    model=PINN().to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)

    for ep in range(1,epochs+1):
        lp_e = 0.0 if ep<=phase1 else lp*min(1.0,(ep-phase1)/100)
        lb_e = 0.0 if ep<=phase1 else lb*min(1.0,(ep-phase1)/100)

        i1=torch.randint(0,N,(8192,),device=DEV)
        Ld=((model(X_tr[i1])-Y_tr[i1])**2*Wd[i1].unsqueeze(-1)).sum(dim=-1).mean()
        total=Ld

        if lp_e>0:
            i2=torch.randint(0,N,(2048,),device=DEV)
            total=total+continuity_loss(model,X_tr[i2],norm)*lp_e

        if lb_e>0 and wall is not None:
            nw=min(wall.shape[0],4096)
            wb=wall[torch.randint(0,wall.shape[0],(nw,),device=DEV)]
            total=total+(model(wb)[:,:3]**2).sum(dim=-1).mean()*lb_e

        opt.zero_grad();total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
        opt.step()

    with torch.no_grad():
        ri=r2_mag(model(X_int).cpu().numpy(),Y_int.cpu().numpy(),norm)
        re=r2_mag(model(X_ext).cpu().numpy(),Y_ext.cpu().numpy(),norm)
        pfi=per_field(model(X_int).cpu().numpy(),Y_int.cpu().numpy(),norm)
        pfe=per_field(model(X_ext).cpu().numpy(),Y_ext.cpu().numpy(),norm)
    return ri,re,pfi,pfe


if __name__=="__main__":
    print("Loading...",flush=True)
    data=build_datasets()
    all_res={};t0=time.time()

    # === νt NOT applicable with continuity-only ===
    # Skip νt ablation — requires viscous term

    # === Exp 3: Physics term ablation ===
    print("\n"+"="*50)
    print("Exp 3: Physics term ablation (continuity)")
    print("="*50)
    phys={}
    for name,lp in [("pure_data",0),("data+BC",0.01),("data+cont",0.001),("data+BC+cont",0.001)]:
        lb_v=0.01 if "BC" in name else 0
        t1=time.time()
        print(f"  {name}...",end=" ",flush=True)
        ri,re,pfi,pfe=train_abl(data,lp=lp,lb=lb_v,epochs=1200,phase1=400)
        print(f"R2_int={ri:.4f} R2_ext={re:.4f} | {time.time()-t1:.0f}s")
        phys[name]={'r2i':ri,'r2e':re,'pfi':pfi,'pfe':pfe}
    all_res['physics_ablation']=phys

    # === Exp 8: VOF ablation ===
    print("\n"+"="*50)
    print("Exp 8: VOF ablation (pure data)")
    print("="*50)
    vof_res={}
    for vof in ["filtered","hard","none"]:
        t1=time.time()
        print(f"  VOF={vof}...",end=" ",flush=True)
        ri,re,pfi,pfe=train_abl(data,lp=0,vof_mode=vof,epochs=1200,phase1=10**9)
        print(f"R2_int={ri:.4f} R2_ext={re:.4f} | {time.time()-t1:.0f}s")
        vof_res[vof]={'r2i':ri,'r2e':re,'pfi':pfi,'pfe':pfe}
    all_res['vof_ablation']=vof_res

    # === Exp 2 simplified: continuity weight sensitivity ===
    print("\n"+"="*50)
    print("Exp: continuity λ sensitivity")
    print("="*50)
    lp_res={}
    for lp in [0,1e-5,1e-4,0.001,0.01,0.1]:
        t1=time.time()
        print(f"  lp={lp:.5f}...",end=" ",flush=True)
        ri,re,pfi,pfe=train_abl(data,lp=lp,epochs=1200,phase1=400)
        print(f"R2_int={ri:.4f} | {time.time()-t1:.0f}s")
        lp_res[str(lp)]={'r2i':ri,'r2e':re}
    all_res['continuity_sweep']=lp_res

    # === l_m sensitivity (for continuity, l_m not used; skip) ===
    all_res['lm_sensitivity']={'note':'l_m only affects momentum term, skipped for continuity-only physics'}

    # === Convergence analysis (loss curves from training) ===
    print("\n"+"="*50)
    print("Exp 6: Convergence (loss curve for pure data)")
    print("="*50)
    torch.manual_seed(42)
    model=PINN().to(data['X_train'].device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    X_int,Y_int=data['X_interp'],data['Y_interp']
    N=len(X_tr);DEV=X_tr.device;norm=data['norm']
    loss_curve=[]
    for ep in range(1,2001):
        idx=torch.randint(0,N,(8192,),device=DEV)
        loss=((model(X_tr[idx])-Y_tr[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
        if ep%50==0:
            with torch.no_grad():ri=r2_mag(model(X_int).cpu().numpy(),Y_int.cpu().numpy(),norm)
            loss_curve.append({'ep':ep,'loss':float(loss.item()),'r2':ri})
    all_res['convergence']=loss_curve
    print(f"  Final ep=2000: R2_int={loss_curve[-1]['r2']:.4f}")

    # Save
    total_t=(time.time()-t0)/60
    out={'results':all_res,'time_min':total_t,'baseline_8x384':0.983}
    with open(os.path.join(RESULT_DIR,'complete_ablations.json'),'w') as f:
        json.dump(out,f,indent=2)
    print(f"\nALL DONE: {total_t:.1f} min")
