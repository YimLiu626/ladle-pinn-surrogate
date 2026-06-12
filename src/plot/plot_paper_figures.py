"""
Generate all paper figures from saved experiment results.
Output: paper/figures/*.pdf (vector) + paper/figures/*.png (raster for 3D)
"""
import sys,os,json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import build_datasets
import csv

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'paper','figures')
os.makedirs(OUT_DIR,exist_ok=True)

# Style
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                      'legend.fontsize':9,'figure.dpi':150,'savefig.dpi':300,
                      'savefig.bbox':'tight','font.family':'serif'})
COLS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR,name), pad_inches=0.05)
    print(f'  Saved: {name}')
    plt.close()

# ============================================================
# Fig 2: VOF phase stratification
# ============================================================
print('Fig 2: VOF stratification')
Qs=['40','60','80','100','120']
fig,ax=plt.subplots(1,1,figsize=(5,5))
for qi,q in enumerate(Qs):
    with open(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv') as f:
        r=csv.DictReader(f,skipinitialspace=True)
        zs,sts=[],[]
        for row in r: zs.append(float(row['z-coordinate']));sts.append(float(row['steel-vof']))
    zs=np.array(zs);sts=np.array(sts)
    zb=np.linspace(0,2.6,50);means=[sts[np.abs(zs-zc)<0.05].mean() for zc in zb]
    ax.plot(means,zb,'-',color=COLS[qi],alpha=0.7+0.3*qi/4,lw=1.5,label=f'Q={q}')
ax.axhline(1.85,color='k',ls='--',alpha=0.4)
ax.set_xlabel(r'$\langle\alpha_{\rm steel}\rangle$');ax.set_ylabel('z (m)')
ax.set_title('Vertical steel fraction');ax.legend(ncol=3,fontsize=8)
ax.set_xlim(0,1.05)
savefig('fig2_vof_stratification.pdf')

# ============================================================
# Fig 3: Lambda sensitivity
# ============================================================
print('Fig 3: Lambda sensitivity')
sweep_path='E:/Submission-EACFM/results/complete_ablations.json'
if os.path.exists(sweep_path):
    with open(sweep_path) as f: data=json.load(f)
    lp_res=data['results']['continuity_sweep']
    lps=[float(k) for k in lp_res.keys()];ris=[lp_res[k]['r2i'] for k in lp_res.keys()]
    fig,ax=plt.subplots(1,1,figsize=(5,4))
    ax.semilogx(lps,ris,'o-',color=COLS[0],ms=6)
    ax.axhline(lp_res['0']['r2i'],color='gray',ls='--',alpha=0.5,label='Pure data baseline')
    ax.set_xlabel(r'$\lambda_{\rm cont}$');ax.set_ylabel(r'$R^2_{\rm int}$')
    ax.set_title('Continuity weight sensitivity')
    ax.legend(fontsize=9)
    savefig('fig3_lambda_sweep.pdf')

# ============================================================
# Fig 4: Physics term ablation
# ============================================================
print('Fig 4: Physics ablation')
phys_names=['Pure data','+ BC','+ Continuity','+ BC + Cont']
phys_r2=[0.859,0.854,0.873,0.854]
fig,ax=plt.subplots(1,1,figsize=(5,4))
bars=ax.bar(phys_names,phys_r2,color=COLS[:4],width=0.6)
for b,r in zip(bars,phys_r2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.002,f'{r:.3f}',ha='center',fontsize=10)
ax.set_ylabel(r'$R^2_{\rm int}$')
ax.set_title('Physics term ablation')
ax.set_ylim(0.84,0.88)
savefig('fig4_physics_ablation.pdf')

# ============================================================
# Fig 5: Baseline comparison
# ============================================================
print('Fig 5: Baseline comparison')
baseline_names=['MLP 8x384\n(Ours)','MLP 6x128','PINN\n+cont','DeepONet']
baseline_r2=[0.983,0.904,0.873,-0.72]
baseline_params=['1.04M','84K','84K','58K']
fig,ax=plt.subplots(1,1,figsize=(6,4))
colors=[COLS[0],COLS[1],COLS[2],COLS[4]]
bars=ax.bar(baseline_names,baseline_r2,color=colors,width=0.6)
ax.axhline(0,color='gray',lw=0.8)
for b,r in zip(bars,baseline_r2):
    y=r+0.03 if r>0 else r-0.15
    ax.text(b.get_x()+b.get_width()/2,y,f'{r:.3f}',ha='center',fontsize=10)
ax.set_ylabel(r'$R^2$')
ax.set_title('Baseline comparison')
savefig('fig5_baselines.pdf')

# ============================================================
# Fig 6: Cross-position generalization
# ============================================================
print('Fig 6: Cross-position')
cross_path='E:/Submission-EACFM/results/cross_position.json'
if os.path.exists(cross_path):
    with open(cross_path) as f: cp=json.load(f)
    labels=['Center\nint','Center\next','Side\nQ40','Side\nQ80','Side\nQ120']
    m0_r2=[cp['center_baseline'],cp['center_baseline'],
           cp['side_Q40']['blind'],cp['side_Q80']['blind'],cp['side_Q120']['blind']]
    joint_r2=[cp['joint_center'],cp['joint_center'],
              cp['joint_Q40'],cp['joint_Q80'],cp['joint_Q120']]
    fig,ax=plt.subplots(1,1,figsize=(7,4))
    x=np.arange(len(labels));w=0.35
    b1=ax.bar(x-w/2,m0_r2,w,color=COLS[1],label='Center-only ($m=0$)')
    b2=ax.bar(x+w/2,joint_r2,w,color=COLS[0],label='Joint ($m\\in\\{0,1\\}$)')
    ax.axhline(0,color='gray',ls='-',lw=0.8)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    ax.set_ylabel(r'$R^2$');ax.legend();ax.set_title('Cross-position generalization')
    for bs in [b1,b2]:
        for b in bs:
            v=b.get_height()
            ax.text(b.get_x()+b.get_width()/2,v+0.03 if v>0 else v-0.12,f'{v:.2f}',ha='center',fontsize=8)
    savefig('fig6_cross_position.pdf')

# ============================================================
# Fig 7: Convergence curves
# ============================================================
print('Fig 7: Convergence')
with open(sweep_path) as f: data=json.load(f)
conv=data['results']['convergence']
eps=[c['ep'] for c in conv];r2s=[c['r2'] for c in conv]
fig,ax=plt.subplots(1,1,figsize=(5,4))
ax.plot(eps,r2s,'-',color=COLS[0],lw=1.5)
ax.set_xlabel('Epoch');ax.set_ylabel(r'$R^2_{\rm int}$')
ax.set_title('Convergence (6x128, pure data)')
savefig('fig7_convergence.pdf')

# ============================================================
# Fig 8: Extrapolation per-Q R2
# ============================================================
print('Fig 8: Extrapolation per Q')
q_labels=['Q40','Q60','Q80','Q100','Q120']
q_types=['Train','Train','Train','Extrap','Extrap']
q_r2=[0.975,0.983,0.987,0.988,0.984]
fig,ax=plt.subplots(1,1,figsize=(5,4))
colors_q=[COLS[0] if t=='Train' else COLS[1] for t in q_types]
ax.bar(q_labels,q_r2,color=colors_q,width=0.6)
ax.axhline(max(q_r2[:3]),color=COLS[0],ls='--',alpha=0.5,label=f'Train mean={np.mean(q_r2[:3]):.3f}')
ax.set_ylabel(r'$R^2$');ax.set_title('Per-Q performance (8x384)')
ax.legend(fontsize=9)
ax.set_ylim(0.96,1.0)
savefig('fig8_perQ.pdf')

# ============================================================
# Fig 9: Multi-seed box plot
# ============================================================
print('Fig 9: Multi-seed')
with open('E:/Submission-EACFM/results/final_results.json') as f: ms=json.load(f)
seeds_vals=ms['multi_seed']['values']
fig,ax=plt.subplots(1,1,figsize=(4,4))
bp=ax.boxplot([seeds_vals],labels=['8x384 joint'],widths=0.4)
ax.scatter([1]*len(seeds_vals),seeds_vals,color=COLS[0],s=40,zorder=5)
ax.set_ylabel(r'$R^2_{\rm int}$');ax.set_title(f'Multi-seed: {np.mean(seeds_vals):.4f}$\\pm${np.std(seeds_vals):.4f}')
savefig('fig9_multiseed.pdf')

# ============================================================
# Fig 10: VOF ablation
# ============================================================
print('Fig 10: VOF ablation')
with open(sweep_path) as f: dd=json.load(f)
vof=dd['results']['vof_ablation']
vof_names=['Filtered\n($\\alpha_s$>0.01)','Hard mask\n($\\alpha_s$>0.5)']
vof_r2=[vof['filtered']['r2i'],vof['hard']['r2i']]
fig,ax=plt.subplots(1,1,figsize=(5,4))
bars=ax.bar(vof_names,vof_r2,color=[COLS[0],COLS[4]],width=0.5)
for b,r in zip(bars,vof_r2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{r:.3f}',ha='center')
ax.set_ylabel(r'$R^2_{\rm int}$');ax.set_title('VOF strategy ablation')
savefig('fig10_vof_ablation.pdf')

# ============================================================
# Fig 1: Framework schematic — placeholder (user provides)
# ============================================================
print('Fig 1: Framework schematic — user to provide diagram')

# ============================================================
# Fig 11: CFD velocity contours (YZ center plane, Q=60,80,100)
# ============================================================
print('Fig 11: CFD velocity contours')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for idx,q in enumerate(['60','80','100']):
    with open(f'E:/Submission-EACFM/data/cfd/bottom/q{q}-1.csv') as f:
        r=csv.DictReader(f,skipinitialspace=True)
        ys,zs,mags=[],[],[]
        for row in r:
            if abs(float(row['x-coordinate']))<0.02:
                ys.append(float(row['y-coordinate']))
                zs.append(float(row['z-coordinate']))
                mags.append(np.sqrt(float(row['x-velocity'])**2+float(row['y-velocity'])**2+float(row['z-velocity'])**2))
    sc=axes[idx].scatter(ys,zs,c=mags,s=0.3,cmap='jet',vmax=0.15,rasterized=True)
    axes[idx].axhline(1.85,color='k',ls='--',alpha=0.4)
    axes[idx].set_xlabel('Y (m)');axes[idx].set_ylabel('Z (m)')
    axes[idx].set_title(f'Q={q} NL/min')
    if idx==2: plt.colorbar(sc,ax=axes[idx],label='|U| (m/s)')
plt.suptitle('CFD velocity field (YZ center plane)')
plt.tight_layout()
savefig('fig11_cfd_contours.pdf')

# ============================================================
# Fig 12: Side-blowing YZ at x=0.47
# ============================================================
print('Fig 12: Side-blowing contours')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for idx,q in enumerate(['40','80','120']):
    with open(f'E:/Submission-EACFM/data/cfd/side/side-q{q}.csv') as f:
        r=csv.DictReader(f,skipinitialspace=True)
        ys,zs,mags=[],[],[]
        for row in r:
            if abs(float(row['x-coordinate'])-0.47)<0.03:
                ys.append(float(row['y-coordinate']))
                zs.append(float(row['z-coordinate']))
                mags.append(np.sqrt(float(row['x-velocity'])**2+float(row['y-velocity'])**2+float(row['z-velocity'])**2))
    sc=axes[idx].scatter(ys,zs,c=mags,s=0.3,cmap='jet',vmax=0.15,rasterized=True)
    axes[idx].axhline(1.85,color='k',ls='--',alpha=0.4)
    axes[idx].set_xlabel('Y (m)');axes[idx].set_ylabel('Z (m)')
    axes[idx].set_title(f'Ecc. Q={q} (off-center plane)')
    if idx==2: plt.colorbar(sc,ax=axes[idx],label='|U| (m/s)')
plt.suptitle('CFD velocity field (eccentric blowing, x=0.47 m)')
plt.tight_layout()
savefig('fig12_eccentric_contours.pdf')

print(f'\nAll figures saved to: {OUT_DIR}')
print(f'Total: {len(os.listdir(OUT_DIR))} files')
