import pickle, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from srcfit import *
import os
D=os.environ.get('FIT_DATA','.')+'/'; os.makedirs('results',exist_ok=True)
kern=Kernel(D+'Kl-ppiplus.dat'); C=np.loadtxt(D+'65.dat'); qs=make_qgrid()
nz=np.where(np.linalg.norm(qs,axis=1)>0)[0]  # all q
NS=20000
plt.rcParams.update({'xtick.top':True,'ytick.right':True,'xtick.direction':'in','ytick.direction':'in',
                     'xtick.minor.visible':True,'ytick.minor.visible':True,'font.size':11})
srcs={}
for f in ['0.54','0.6','0.65','0.69','0.73','0.76','0.79']:
    srcs['ppi-'+f]=np.loadtxt(D+f'ppi-{f}.dat')[:NS]
gauss,hg=pickle.load(open('results/fit_gauss.pkl','rb')); nvp,hn=pickle.load(open('results/fit_realnvp.pkl','rb'))
init=Source([4,0,-.5],np.diag([80.,30,30]))
models={'initial guess (notebook)':sample_source(init,NS),'Gaussian fit':sample_source(gauss,NS),'RealNVP fit':sample_source(nvp,NS)}
def crms(r):
    c=correlation(kern,qs[nz],r); return float(np.sqrt(np.mean((c-C[nz])**2))), c
def w1(a,b): return float(np.mean(np.abs(np.sort(a)-np.sort(b))))
def skew(x): x=x-x.mean(); return float((x**3).mean()/x.std()**3)
ref=srcs['ppi-0.65']
rows=[]
allsrc={**models,**srcs}
corr={}
for name,r in allsrc.items():
    rm,c=crms(r); corr[name]=c
    rn=np.linalg.norm(r,axis=1)
    rows.append(dict(name=name,C_rms=rm,mean=r.mean(0).round(2).tolist(),std=r.std(0).round(2).tolist(),
        cov_xz=float(np.cov(r.T)[0,2]),skew=[round(skew(r[:,i]),2) for i in range(3)],
        W1_x=w1(r[:,0],ref[:,0]),W1_y=w1(r[:,1],ref[:,1]),W1_z=w1(r[:,2],ref[:,2]),W1_r=w1(rn,np.linalg.norm(ref,axis=1))))
json.dump(rows,open('results/metrics.json','w'),indent=1)
print(f"{'source':26s} {'C rms':>8s} {'mean':>22s} {'std':>20s} {'cov_xz':>7s} {'skew':>18s}   W1(x,y,z,|r|) to ppi-0.65")
for x in rows:
    print(f"{x['name']:26s} {x['C_rms']:8.5f} {str(x['mean']):>22s} {str(x['std']):>20s} {x['cov_xz']:7.1f} {str(x['skew']):>18s}   {x['W1_x']:.2f} {x['W1_y']:.2f} {x['W1_z']:.2f} {x['W1_r']:.2f}")
# ---- figures
col={'ppi-0.65':'k','Gaussian fit':'tab:blue','RealNVP fit':'tab:red'}
fig,ax=plt.subplots(2,3,figsize=(13,7.5))
names=['x (out)','y (side)','z (long)']
for i in range(3):
    a=ax.flat[i]; lo,hi=np.percentile(ref[:,i],[0.1,99.9]); b=np.linspace(lo-5,hi+5,90)
    for n,r in [('ppi-0.65',ref),('Gaussian fit',models['Gaussian fit']),('RealNVP fit',models['RealNVP fit'])]:
        a.hist(r[:,i],bins=b,density=True,histtype='step',lw=1.8 if n!='ppi-0.65' else 2.2,color=col[n],label=n if i==0 else None,ls='-' if n!='ppi-0.65' else '-')
    a.set_xlabel(names[i]+' [fm]'); a.set_ylabel('density'); a.set_yscale('log'); a.set_ylim(1e-5,None)
ax.flat[0].legend(frameon=False,fontsize=9)
a=ax.flat[3]; b=np.linspace(0,60,90)
for n,r in [('ppi-0.65',ref),('Gaussian fit',models['Gaussian fit']),('RealNVP fit',models['RealNVP fit'])]:
    a.hist(np.linalg.norm(r,axis=1),bins=b,density=True,histtype='step',lw=1.8,color=col[n])
a.set_xlabel('|r| [fm]'); a.set_ylabel('density')
for k,(n,r) in enumerate([('Gaussian fit',models['Gaussian fit']),('RealNVP fit',models['RealNVP fit'])]):
    a=ax.flat[4+k]
    H,xe,ze=np.histogram2d(ref[:,0],ref[:,2],bins=80,range=[[-30,45],[-35,35]],density=True)
    Hm,_,_=np.histogram2d(r[:,0],r[:,2],bins=80,range=[[-30,45],[-35,35]],density=True)
    lv=np.max(H)*np.array([.01,.05,.2,.5,.8])
    X,Z=np.meshgrid((xe[1:]+xe[:-1])/2,(ze[1:]+ze[:-1])/2,indexing='ij')
    a.contour(X,Z,H,levels=lv,colors='k',linewidths=1.5); a.contour(X,Z,Hm,levels=lv,colors=col[n],linewidths=1.5,linestyles='--')
    a.set_xlabel('x [fm]'); a.set_ylabel('z [fm]'); a.set_title(n+' (dashed) vs ppi-0.65 (solid)',fontsize=10)
fig.tight_layout(); fig.savefig('results/fig_source_comparison.png',dpi=140)
# correlation function figure
fig,ax=plt.subplots(1,3,figsize=(14,4.2))
qn=np.linalg.norm(qs[nz],axis=1)
for n,c in [('Gaussian fit',corr['Gaussian fit']),('RealNVP fit',corr['RealNVP fit']),('ppi-0.65',corr['ppi-0.65'])]:
    ax[0].hist(c-C[nz],bins=np.linspace(-.05,.05,101),histtype='step',lw=1.8,color=col[n],label=f"{n}: rms {np.sqrt(np.mean((c-C[nz])**2)):.4f}")
ax[0].set_xlabel('C_model(q) - C_65(q)'); ax[0].set_ylabel('number of q points'); ax[0].legend(frameon=False,fontsize=8)
line=np.where((qs[nz][:,1]==-5)&(qs[nz][:,2]==-5))[0]
ax[1].plot(qs[nz][line,0],C[nz][line],'ko',ms=4,label='65.dat')
for n in ['Gaussian fit','RealNVP fit']: ax[1].plot(qs[nz][line,0],corr[n][line],color=col[n],label=n)
ax[1].set_xlabel('q_x [MeV/c]'); ax[1].set_ylabel('C(q_x, -5, -5)'); ax[1].legend(frameon=False,fontsize=8)
line=np.where((qs[nz][:,0]==-5)&(qs[nz][:,1]==-5))[0]
ax[2].plot(qs[nz][line,2],C[nz][line],'ko',ms=4)
for n in ['Gaussian fit','RealNVP fit']: ax[2].plot(qs[nz][line,2],corr[n][line],color=col[n])
ax[2].set_xlabel('q_z [MeV/c]'); ax[2].set_ylabel('C(-5, -5, q_z)')
for a in ax: a.minorticks_on()
fig.tight_layout(); fig.savefig('results/fig_correlation_fit.png',dpi=140)
fig,a=plt.subplots(figsize=(6,4))
sm=lambda h:np.convolve(h,np.ones(50)/50,mode='valid')
a.plot(sm(hg),color=col['Gaussian fit'],label='Gaussian'); a.plot(sm(hn),color=col['RealNVP fit'],label='RealNVP')
a.set_xlabel('training step'); a.set_ylabel('loss  ||C_model - C||  (N=1000 samples)'); a.legend(frameon=False); a.minorticks_on()
fig.tight_layout(); fig.savefig('results/fig_loss.png',dpi=140)
