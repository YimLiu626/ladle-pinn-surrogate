"""Generate 4x3 error figure: 3 rows PINN + 1 row POD"""
import sys;sys.path.insert(0,'E:/Submission-EACFM/src')
import torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_loader import normalize_x, normalize_y
import csv
from scipy.interpolate import RBFInterpolator

class BigMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers=[5]+[384]*8+[4];net=[]
        for i in range(len(layers)-1):
            net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i<len(layers)-2:net.append(torch.nn.SiLU())
        self.net=torch.nn.Sequential(*net)
    def forward(self,x):return self.net(x)

ckpt=torch.load('E:/Submission-EACFM/models/final_best.pt',map_location='cuda',weights_only=False)
model=BigMLP().to('cuda');model.load_state_dict(ckpt['model_state']);model.eval()
jckpt=torch.load('E:/Submission-EACFM/models/joint_model.pt',map_location='cuda',weights_only=False)
jmodel=BigMLP().to('cuda');jmodel.load_state_dict(jckpt['model_state']);jmodel.eval()
norm=jckpt['norm'];DEV='cuda'

def load_cfd(path):
    xs,ys,zs,us,vs,ws=[],[],[],[],[],[]
    with open(path) as f:
        rr=csv.DictReader(f,skipinitialspace=True)
        for row in rr:
            xs.append(float(row['x-coordinate']));ys.append(float(row['y-coordinate']))
            zs.append(float(row['z-coordinate']))
            us.append(float(row['x-velocity']));vs.append(float(row['y-velocity']));ws.append(float(row['z-velocity']))
    mag=np.sqrt(np.array(us)**2+np.array(vs)**2+np.array(ws)**2)
    return mag,np.array(xs),np.array(ys),np.array(zs)

def get_pinn_err(path,q,m,md):
    mag_cfd,xs,ys,zs=load_cfd(path);n=len(xs)
    Xr=np.column_stack([xs,ys,zs,np.full(n,q),np.full(n,m)])
    Xt=torch.tensor(normalize_x(Xr,norm),dtype=torch.float32,device=DEV)
    with torch.no_grad():pd=md(Xt).cpu().numpy()
    for j,nm in enumerate(['u','v','w','p']):
        ks,km=f'{nm}_std',f'{nm}_mean';pd[:,j]=pd[:,j]*norm[ks]+norm[km]
    return np.abs(np.sqrt((pd[:,:3]**2).sum(1))-mag_cfd),xs,ys,zs

# Build POD
mags=[load_cfd(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv')[0] for q in[40,60,80]]
X=np.vstack(mags);X_mean=X.mean(axis=0);Xc=X-X_mean
U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
modes=Vt[:2];coeffs=U[:,:2]*S[:2]
q_arr=np.array([40,60,80]).reshape(-1,1)
rbfs=[RBFInterpolator(q_arr,coeffs[:,i],kernel='thin_plate_spline') for i in range(2)]

def get_pod_err(q):
    c_pred=np.array([rbf(np.array([[q]]))[0] for rbf in rbfs])
    pod_mag=X_mean+modes.T@c_pred
    cfd_mag,xs,ys,zs=load_cfd(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv')
    return np.abs(pod_mag-cfd_mag),xs,ys,zs

levels=np.linspace(0,0.03,16)
colors=plt.cm.YlOrRd(np.linspace(0.15,1,15))
cmap=ListedColormap(colors);norm_cb=BoundaryNorm(levels,len(colors))

fig=plt.figure(figsize=(24,30))
gs=fig.add_gridspec(4,3,hspace=0.12,wspace=0.02)

def draw_ladle(ax):
    theta=np.linspace(0,2*np.pi,60);r_bot=0.92;r_top=1.04
    for z_val,r in[(0,r_bot),(1.85,r_top)]:
        ax.plot(r*np.cos(theta),r*np.sin(theta),z_val,color='gray',lw=0.5,alpha=0.4)
    for th in[0,np.pi/2,np.pi,3*np.pi/2]:
        ax.plot([r_bot*np.cos(th),r_top*np.cos(th)],[r_bot*np.sin(th),r_top*np.sin(th)],[0,1.85],color='gray',lw=0.5,alpha=0.4)

def plot3d(ax,xs,ys,zs,err,title):
    st=zs<1.88;n_plot=min(20000,st.sum())
    idx_s=np.random.default_rng(42).choice(st.sum(),n_plot,replace=False)
    ax.scatter(xs[st][idx_s],ys[st][idx_s],zs[st][idx_s],c=err[st][idx_s],s=0.1,cmap=cmap,norm=norm_cb,alpha=0.5,rasterized=True)
    draw_ladle(ax)
    ax.view_init(elev=20,azim=-30);ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.set_title(title,fontweight='bold',fontsize=10);ax.set_zlim(0,1.88)

# Rows 1-3: PINN (same as before)
for idx,(q,lbl) in enumerate([(40,'(a)'),(60,'(b)'),(80,'(c)')]):
    err,xs,ys,zs=get_pinn_err(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv',q,0,model)
    plot3d(fig.add_subplot(gs[0,idx],projection='3d'),xs,ys,zs,err,f'{lbl} PINN Center Q={q}')

for idx,(q,lbl) in enumerate([(100,'(d)'),(120,'(e)')]):
    err,xs,ys,zs=get_pinn_err(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv',q,0,model)
    plot3d(fig.add_subplot(gs[1,idx],projection='3d'),xs,ys,zs,err,f'{lbl} PINN Center Q={q}')

ax=fig.add_subplot(gs[1,2])
ec,xsc,ysc,zsc=get_pinn_err('E:/Submission-EACFM/data/cfd/bottom/q120-1.csv',120,0,model)
es,xss,yss,zss=get_pinn_err('E:/Submission-EACFM/data/cfd/side/side-q120.csv',120,1,jmodel)
ax.hist(ec[zsc<1.88],bins=60,density=True,color='#2c3e50',alpha=0.5,label='Center Q120',range=(0,0.03))
ax.hist(es[zss<1.88],bins=60,density=True,color='#e74c3c',alpha=0.5,label='Ecc. Q120',range=(0,0.03))
ax.set_xlabel('|U| error');ax.set_ylabel('Density')
ax.set_title('(f) PINN Error: Q120',fontweight='bold',fontsize=10);ax.legend(fontsize=8)

for idx,(q,lbl) in enumerate([(40,'(g)'),(80,'(h)'),(120,'(i)')]):
    err,xs,ys,zs=get_pinn_err(f'E:/Submission-EACFM/data/cfd/side/side-q{q}.csv',q,1,jmodel)
    plot3d(fig.add_subplot(gs[2,idx],projection='3d'),xs,ys,zs,err,f'{lbl} PINN Ecc. Q={q}')

# Row 4: POD
err,xs,ys,zs=get_pod_err(120)
plot3d(fig.add_subplot(gs[3,0],projection='3d'),xs,ys,zs,err,'(j) POD Center Q120')

cfd_mag,xs,ys,zs=load_cfd('E:/Submission-EACFM/data/cfd/side/side-q120.csv')
c_pred=np.array([rbf(np.array([[120]]))[0] for rbf in rbfs])
pod_mag=X_mean+modes.T@c_pred
err2=np.abs(pod_mag-cfd_mag)
plot3d(fig.add_subplot(gs[3,1],projection='3d'),xs,ys,zs,err2,'(k) POD blind Ecc. Q120')

ax=fig.add_subplot(gs[3,2])
ec2,xsc2,ysc2,zsc2=get_pod_err(120)
ax.hist(ec2[zsc2<1.88],bins=60,density=True,color='#1abc9c',alpha=0.5,label='POD Center Q120',range=(0,0.03))
ax.hist(err2[zs<1.88],bins=60,density=True,color='#e74c3c',alpha=0.5,label='POD blind Ecc. Q120',range=(0,0.03))
ax.set_xlabel('|U| error');ax.set_ylabel('Density')
ax.set_title('(l) POD Error: Q120',fontweight='bold',fontsize=10);ax.legend(fontsize=8)

cbar_ax=fig.add_axes([0.93,0.08,0.012,0.88])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm_cb,cmap=cmap),cax=cbar_ax,ticks=np.linspace(0,0.03,7))
cb.ax.set_yticklabels([f'{x:.3f}' for x in np.linspace(0,0.03,7)],fontsize=8)
cb.set_label('|U| error (m/s)',fontsize=14)

fig.text(0.03,0.90,'PINN\nCenter',fontsize=14,fontweight='bold',rotation=90,va='center',color='#2c3e50')
fig.text(0.03,0.64,'PINN\nCenter',fontsize=14,fontweight='bold',rotation=90,va='center',color='#2c3e50')
fig.text(0.03,0.38,'PINN\nEcc.',fontsize=14,fontweight='bold',rotation=90,va='center',color='#2c3e50')
fig.text(0.03,0.12,'POD',fontsize=14,fontweight='bold',rotation=90,va='center',color='#1abc9c')

fig.text(0.5,0.998,'3D prediction error: PINN vs POD',ha='center',va='top',fontsize=16,fontweight='bold')
plt.subplots_adjust(top=0.97,left=0.06,right=0.92)
plt.savefig('E:/Submission-EACFM/pic/fig_3d_error.png',dpi=600,bbox_inches='tight')
plt.savefig('E:/Submission-EACFM/paper/figures/fig_3d_error.pdf',dpi=300,bbox_inches='tight')
plt.close()
print('4x3 error with POD row saved')
