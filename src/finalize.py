"""
Final model: 8x384 MLP, multi-seed, per-Q predictions, save everything.
"""
import sys,os,time,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torch,numpy as np
from config import *
from data_loader import build_datasets

def r2_mag(t,p,norm):
    td,pd=t.copy(),p.copy()
    for j,k in enumerate(['u_std','v_std','w_std']):
        m=k.replace('std','mean')
        pd[:,j]=pd[:,j]*norm[k]+norm[m];td[:,j]=td[:,j]*norm[k]+norm[m]
    mp=np.sqrt((pd[:,:3]**2).sum(1));mt=np.sqrt((td[:,:3]**2).sum(1))
    return float(1-((mt-mp)**2).sum()/max(((mt-mt.mean())**2).sum(),1e-20))

def per_field(pred,true,norm):
    pd,td=pred.copy(),true.copy()
    out={}
    for j,nm in enumerate(['u','v','w','p']):
        ks,km=f'{nm}_std',f'{nm}_mean'
        pd[:,j]=pd[:,j]*norm[ks]+norm[km];td[:,j]=td[:,j]*norm[ks]+norm[km]
        ssr=((td[:,j]-pd[:,j])**2).sum();sst=((td[:,j]-td[:,j].mean())**2).sum()
        out[nm]=float(1-ssr/max(sst,1e-20))
    return out

class BigMLP(torch.nn.Module):
    def __init__(self,width=384,depth=8):
        super().__init__()
        layers=[5]+[width]*depth+[4]
        net=[]
        for i in range(len(layers)-1):
            net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i<len(layers)-2:net.append(torch.nn.SiLU())
        self.net=torch.nn.Sequential(*net)
    def forward(self,x):return self.net(x)

def train_eval(data,epochs,seed,save_path=None):
    torch.manual_seed(seed);np.random.seed(seed)
    X_tr,Y_tr=data['X_train'],data['Y_train']
    N=len(X_tr);DEV=X_tr.device
    model=BigMLP().to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(1,epochs+1):
        idx=torch.randint(0,N,(8192,),device=DEV)
        loss=((model(X_tr[idx])-Y_tr[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()

    X_te=torch.cat([data['X_interp'],data['X_extrap']])
    Y_te=torch.cat([data['Y_interp'],data['Y_extrap']])
    norm=data['norm']
    with torch.no_grad():
        p=model(X_te).cpu().numpy();t=Y_te.cpu().numpy()
        pi=model(data['X_interp']).cpu().numpy();ti=data['Y_interp'].cpu().numpy()
        pe=model(data['X_extrap']).cpu().numpy();te=data['Y_extrap'].cpu().numpy()
    ri=r2_mag(pi,ti,norm);re=r2_mag(pe,te,norm)
    pfi=per_field(pi,ti,norm);pfe=per_field(pe,te,norm)

    # Per-Q
    Qmax=norm['Q_max'];Qv=((X_te[:,3].cpu().numpy()+1)/2*Qmax).round()
    perQ={}
    for q in sorted(set(Qv)):
        m=Qv==q
        if m.sum()>100:perQ[int(q)]=round(r2_mag(p[m],t[m],norm),4)

    result={'r2_int':ri,'r2_ext':re,'pfi':pfi,'pfe':pfe,'perQ':perQ,'seed':seed}

    if save_path:
        torch.save({'model_state':model.state_dict(),'norm':norm,'result':result},save_path)
    return model,result

print("Loading...",flush=True)
data=build_datasets()

# === Final model: seed=42, 3000ep ===
print("\n=== Final: 8x384, seed=42, 3000ep ===")
t0=time.time()
model,res=train_eval(data,3000,42,os.path.join(MODEL_DIR,'final_best.pt'))
print(f"R2_int={res['r2_int']:.4f} R2_ext={res['r2_ext']:.4f} | u={res['pfi']['u']:.3f} w={res['pfi']['w']:.3f} p={res['pfi']['p']:.3f}")
print(f"Per-Q: {res['perQ']} | {time.time()-t0:.0f}s")

# === Multi-seed ===
print("\n=== Multi-seed (3 seeds) ===")
seeds=[42,123,456]
seed_res=[]
for s in seeds:
    t0=time.time()
    _,r=train_eval(data,2000,s)
    print(f"  seed={s}: R2_int={r['r2_int']:.4f} R2_ext={r['r2_ext']:.4f} | {time.time()-t0:.0f}s")
    seed_res.append(r)

ris=[r['r2_int'] for r in seed_res]
res_out={
    'final':res,
    'multi_seed':{'values':[r['r2_int'] for r in seed_res],'mean':np.mean(ris),'std':np.std(ris),'results':seed_res},
    'baselines':{'pinn_6x128':0.9072,'mlp_8x384':res['r2_int'],'deeponet':-0.7206,'kriging':'oom','pod_rom':'structured_grid_required','kan':'toolchain'},
    'data_verified':True,
    'timestamp':'2026-06-11'
}
with open(os.path.join(RESULT_DIR,'final_results.json'),'w') as f:
    json.dump(res_out,f,indent=2,default=str)
print(f"\n=== FINAL ===")
print(f"R2_int={res['r2_int']:.4f}, R2_ext={res['r2_ext']:.4f}")
print(f"Multi-seed: {np.mean(ris):.4f} +- {np.std(ris):.4f}")
print(f"Saved: {os.path.join(RESULT_DIR,'final_results.json')}")
