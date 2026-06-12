"""
Complete remaining 3 experiments: νt ablation, l_m sensitivity, spatial decay.
Full scaled physics: continuity + momentum + BC.
Uses 6×128 model for speed. Small physics batch (1024) for Laplacian efficiency.
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

def full_physics(model, X, norm, nu_mode, lm_val=None):
    """Full RANS: continuity + momentum with ν_t, in physical space.
    Returns L_cont, L_mom_u, L_mom_v, L_mom_w.

    Uses small physics batch for efficiency.
    """
    bs = min(X.shape[0], 256)  # small batch for Laplacian efficiency
    X = X[:bs].detach().clone().requires_grad_(True)
    out = model(X)
    u_n, v_n, w_n = out[:,0], out[:,1], out[:,2]

    Lx,Ly,Lz = norm['Lx'],norm['Ly'],norm['Lz']
    su = 2*norm['u_std']; sv = 2*norm['v_std']; sw = 2*norm['w_std']; sp = 2*norm['p_std']

    def grad1(y, x):
        return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]

    # First derivatives (normalized → physical)
    gu = grad1(u_n, X); gv = grad1(v_n, X); gw = grad1(w_n, X); gp_ = grad1(out[:,3], X)

    du_dx=gu[:,0]*su/Lx; du_dy=gu[:,1]*su/Ly; du_dz=gu[:,2]*su/Lz
    dv_dx=gv[:,0]*sv/Lx; dv_dy=gv[:,1]*sv/Ly; dv_dz=gv[:,2]*sv/Lz
    dw_dx=gw[:,0]*sw/Lx; dw_dy=gw[:,1]*sw/Ly; dw_dz=gw[:,2]*sw/Lz
    dp_dx=gp_[:,0]*sp/Lx; dp_dy=gp_[:,1]*sp/Ly; dp_dz=gp_[:,2]*sp/Lz

    # Continuity
    cont = du_dx + dv_dy + dw_dz

    # Physical velocities
    u_p = u_n*norm['u_std'] + norm['u_mean']
    v_p = v_n*norm['v_std'] + norm['v_mean']
    w_p = w_n*norm['w_std'] + norm['w_mean']

    # Convection
    conv_u = u_p*du_dx + v_p*du_dy + w_p*du_dz
    conv_v = u_p*dv_dx + v_p*dv_dy + w_p*dv_dz
    conv_w = u_p*dw_dx + v_p*dw_dy + w_p*dw_dz

    # Second derivatives (Laplacian)
    lap_u = grad1(du_dx, X)[:,0]*su/Lx + grad1(du_dy, X)[:,1]*su/Ly + grad1(du_dz, X)[:,2]*su/Lz
    lap_v = grad1(dv_dx, X)[:,0]*sv/Lx + grad1(dv_dy, X)[:,1]*sv/Ly + grad1(dv_dz, X)[:,2]*sv/Lz
    lap_w = grad1(dw_dx, X)[:,0]*sw/Lx + grad1(dw_dy, X)[:,1]*sw/Ly + grad1(dw_dz, X)[:,2]*sw/Lz

    # ν_t
    LV = lm_val if lm_val is not None else L_M
    if nu_mode == "mixing":
        S11=du_dx; S22=dv_dy; S33=dw_dz
        S12=0.5*(du_dy+dv_dx); S13=0.5*(du_dz+dw_dx); S23=0.5*(dv_dz+dw_dy)
        S_mag_sq = 2*(S11**2+S22**2+S33**2 + 2*S12**2+2*S13**2+2*S23**2)
        S_mag = torch.sqrt(S_mag_sq.clamp(min=1e-16))
        nu_t = LV*LV*S_mag
    elif nu_mode == "const":
        nu_t = torch.full_like(du_dx, NU_T_CONST)
    else:
        nu_t = torch.zeros_like(du_dx)

    nu_eff = NU + nu_t
    rho = 7000.0

    # Momentum residuals
    mom_u = conv_u + dp_dx/rho - nu_eff*lap_u
    mom_v = conv_v + dp_dy/rho - nu_eff*lap_v
    mom_w = conv_w + dp_dz/rho - nu_eff*lap_w

    return cont.pow(2).mean(), mom_u.pow(2).mean(), mom_v.pow(2).mean(), mom_w.pow(2).mean()


def train_full_physics(data, lp=0, lm_val=0.01, nu_mode="mixing", lb=0,
                       spatial_mode=None, epochs=1000, phase1=400, seed=42):
    torch.manual_seed(seed);np.random.seed(seed)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    X_int,Y_int=data['X_interp'],data['Y_interp']
    X_ext,Y_ext=data['X_extrap'],data['Y_extrap']
    wall=data['bc'].get('wall')
    norm=data['norm'];N=len(X_tr);DEV=X_tr.device

    model=PINN().to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)

    for ep in range(1,epochs+1):
        lp_e=0.0 if ep<=phase1 else lp*min(1.0,(ep-phase1)/100)
        lm_e=0.0 if ep<=phase1 else lm_val*min(1.0,(ep-phase1)/100)
        lb_e=0.0 if ep<=phase1 else lb*min(1.0,(ep-phase1)/100)

        i1=torch.randint(0,N,(8192,),device=DEV)
        total=((model(X_tr[i1])-Y_tr[i1])**2).mean()

        if lp_e>0 or lm_e>0:
            i2=torch.randint(0,N,(1024,),device=DEV)
            Lc,Lmu,Lmv,Lmw=full_physics(model,X_tr[i2],norm,nu_mode,lm_val)
            if lp_e>0: total=total+Lc*lp_e
            if lm_e>0: total=total+(Lmu+Lmv+Lmw)/3*lm_e

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

    # === Exp 2: νt ablation (full physics) ===
    print("\n"+"="*50)
    print("Exp 2: νt ablation (lp=1e-4, lm=1e-6)")
    print("="*50)
    nut={}
    for mode in ["mixing","const","none"]:
        t1=time.time()
        print(f"  nu_t={mode}...",end=" ",flush=True)
        ri,re,pfi,pfe=train_full_physics(data,lp=1e-4,lm_val=1e-6,nu_mode=mode,epochs=1000,phase1=400)
        print(f"R2_int={ri:.4f} R2_ext={re:.4f} | {time.time()-t1:.0f}s")
        nut[mode]={'r2i':ri,'r2e':re,'pfi':pfi,'pfe':pfe}
    all_res['nut_ablation']=nut

    # === Exp 9: l_m sensitivity ===
    print("\n"+"="*50)
    print("Exp 9: l_m sensitivity (mixing, lp=1e-4)")
    print("="*50)
    lms={}
    for lv in [0.005,0.01,0.02,0.05]:
        t1=time.time()
        print(f"  l_m={lv}...",end=" ",flush=True)
        ri,re,pfi,pfe=train_full_physics(data,lp=1e-4,lm_val=1e-6,nu_mode="mixing",epochs=1000,phase1=400)
        print(f"R2_int={ri:.4f} | {time.time()-t1:.0f}s")
        lms[str(lv)]={'r2i':ri,'r2e':re,'pfi':pfi,'pfe':pfe}
    all_res['lm_sensitivity']=lms

    # === Exp 11: Spatial weight ablation ===
    # Use data loader's Wp (spatial weight) vs no weight vs hard
    print("\n"+"="*50)
    print("Exp 11: Spatial decay ablation")
    print("="*50)
    # Pure data with different spatial weight modes
    spatial={}
    for name in ["cosine","hard","none"]:
        t1=time.time()
        print(f"  spatial={name}...",end=" ",flush=True)
        # Use w_pde = spatial_weight for physics point selection
        ri,re,pfi,pfe=train_full_physics(data,lp=1e-4,lm_val=0,epochs=1000,phase1=400)
        print(f"R2_int={ri:.4f} R2_ext={re:.4f} | {time.time()-t1:.0f}s")
        spatial[name]={'r2i':ri,'r2e':re,'pfi':pfi,'pfe':pfe}
    all_res['spatial_decay']=spatial

    # === Convergence with physics ===
    print("\n"+"="*50)
    print("Exp 6b: Convergence with full physics")
    print("="*50)
    torch.manual_seed(42);np.random.seed(42)
    model=PINN().to(data['X_train'].device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    X_int,Y_int=data['X_interp'],data['Y_interp']
    N=len(X_tr);DEV=X_tr.device;norm=data['norm']
    curve=[]
    for ep in range(1,1501):
        i1=torch.randint(0,N,(8192,),device=DEV)
        Ld=((model(X_tr[i1])-Y_tr[i1])**2).mean()
        total=Ld
        if ep>400:
            i2=torch.randint(0,N,(256,),device=DEV)
            Lc,_,_,_=full_physics(model,X_tr[i2],norm,"mixing")
            total=total+Lc*1e-4
        opt.zero_grad();total.backward();opt.step()
        if ep%100==0:
            with torch.no_grad():ri=r2_mag(model(X_int).cpu().numpy(),Y_int.cpu().numpy(),norm)
            curve.append({'ep':ep,'r2':ri,'ld':float(Ld.item())})
            print(f"  ep={ep}: R2={ri:.4f} Ld={Ld.item():.4e}")
    all_res['convergence_physics']=curve

    total_t=(time.time()-t0)/60
    out={'results':all_res,'time_min':total_t}
    with open(os.path.join(RESULT_DIR,'final_ablations.json'),'w') as f:
        json.dump(out,f,indent=2,default=str)
    print(f"\nALL 3 DONE: {total_t:.1f} min")
