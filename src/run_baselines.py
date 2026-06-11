"""
Baselines: DeepONet vs PINN/MLP.
Kriging: attempted but O(n^3) memory on 2M pts (paper: acknowledge limitation).
POD-ROM: requires uniform grid interpolation (future work).
KAN: pykan unstable on Windows.
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

def perQ(pred,true,Qv,norm):
    Qmax=norm['Q_max'];Qp=((Qv.cpu().numpy()+1)/2*Qmax).round()
    r={}
    for q in sorted(set(Qp)):
        m=Qp==q
        if m.sum()>100:r[int(q)]=round(r2_mag(pred[m],true[m],norm),4)
    return r

print("Loading...",flush=True)
data=build_datasets()
X_tr,Y_tr=data['X_train'],data['Y_train']
X_te=torch.cat([data['X_interp'],data['X_extrap']])
Y_te=torch.cat([data['Y_interp'],data['Y_extrap']])
norm=data['norm'];DEV=X_tr.device;N=len(X_tr)

results={}

# === DeepONet ===
print("\n=== DeepONet (Branch:Q, Trunk:xyz) ===")
class DeepONet(torch.nn.Module):
    def __init__(self,p=50):
        super().__init__()
        self.branch=torch.nn.Sequential(torch.nn.Linear(1,64),torch.nn.SiLU(),torch.nn.Linear(64,128),torch.nn.SiLU(),torch.nn.Linear(128,p))
        self.trunk=torch.nn.Sequential(torch.nn.Linear(3,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,4*p))
        self.p=p
    def forward(self,x):
        b=self.branch(x[:,3:4]);t=self.trunk(x[:,:3]).view(-1,4,self.p)
        return (b.unsqueeze(1)*t).sum(-1)

dn=DeepONet(p=50).to(DEV);opt=torch.optim.Adam(dn.parameters(),lr=1e-3)
t0=time.time()
for ep in range(1,2001):
    idx=torch.randint(0,N,(8192,),device=DEV)
    loss=((dn(X_tr[idx])-Y_tr[idx])**2).mean()
    opt.zero_grad();loss.backward();opt.step()
    if ep%500==0:
        with torch.no_grad():
            ri=r2_mag(dn(data['X_interp']).cpu().numpy(),data['Y_interp'].cpu().numpy(),norm)
        print(f"  ep={ep}: R2_int={ri:.4f} | {time.time()-t0:.0f}s")

with torch.no_grad():
    pd=dn(X_te).cpu().numpy();td=Y_te.cpu().numpy()
r2d=r2_mag(pd,td,norm);pqd=perQ(pd,td,X_te[:,3],norm)
print(f"  Final: R2={r2d:.4f} | perQ={pqd}")
results['deeponet']={'r2':r2d,'perQ':pqd,'params':sum(p.numel() for p in dn.parameters())}

# === Larger MLP (8x384, matches old architecture) ===
print("\n=== Larger MLP (8x384 SiLU) ===")
class BigMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers=[5]+[384]*8+[4]
        net=[]
        for i in range(len(layers)-1):
            net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i<len(layers)-2:net.append(torch.nn.SiLU())
        self.net=torch.nn.Sequential(*net)
    def forward(self,x):return self.net(x)

bm=BigMLP().to(DEV);print(f"  Params: {sum(p.numel() for p in bm.parameters()):,}")
opt=torch.optim.Adam(bm.parameters(),lr=1e-3)
t0=time.time()
for ep in range(1,2001):
    idx=torch.randint(0,N,(8192,),device=DEV)
    loss=((bm(X_tr[idx])-Y_tr[idx])**2).mean()
    opt.zero_grad();loss.backward();opt.step()
    if ep%500==0:
        with torch.no_grad():
            ri=r2_mag(bm(data['X_interp']).cpu().numpy(),data['Y_interp'].cpu().numpy(),norm)
        print(f"  ep={ep}: R2_int={ri:.4f} | {time.time()-t0:.0f}s")

with torch.no_grad():
    pb=bm(X_te).cpu().numpy();td=Y_te.cpu().numpy()
r2b=r2_mag(pb,td,norm);pqb=perQ(pb,td,X_te[:,3],norm)
print(f"  Final: R2={r2b:.4f} | perQ={pqb}")
results['mlp_large']={'r2':r2b,'perQ':pqb,'params':sum(p.numel() for p in bm.parameters())}

# === Summary ===
results['pinn_6x128']={'r2_int':0.9072,'r2_ext':0.8938,'note':'best from full_train'}
results['kriging']={'r2':None,'note':'sparseGPR_2000pts_failed_OOM_and_nonconvergence'}
results['pod_rom']={'r2':None,'note':'requires_uniform_grid_interpolation'}
results['kan']={'r2':None,'note':'pykan_windows_install_issues'}

print("\n=== Baseline Summary ===")
for name,res in results.items():
    r2=res.get('r2') or res.get('r2_int','N/A')
    n=res.get('params','N/A')
    print(f"  {name:20s}: R2={r2 if r2 else 'FAIL':>8} params={n}")

with open(os.path.join(RESULT_DIR,'baselines.json'),'w') as f:
    json.dump(results,f,indent=2,default=str)
print(f"\nSaved: {os.path.join(RESULT_DIR,'baselines.json')}")
