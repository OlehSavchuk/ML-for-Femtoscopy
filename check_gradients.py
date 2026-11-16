import numpy as np, time
from srcfit import *
import os
D=os.environ.get('FIT_DATA','.')+'/'
kern=Kernel(D+'Kl-ppiplus.dat'); C=np.loadtxt(D+'65.dat'); qs=make_qgrid()
sel=np.where(np.linalg.norm(qs,axis=1)>0)[0][::8]; q,Cs=qs[sel],C[sel]
rng=np.random.default_rng(0)
# 1) dL/dr
r=rng.normal(size=(200,3))*[8,5,5]+[7,0,-.5]
L,g=loss_and_grad_r(kern,q,Cs,r)
for (n,k) in [(3,0),(10,1),(50,2),(120,0)]:
    h=1e-5; rp=r.copy(); rp[n,k]+=h; rm=r.copy(); rm[n,k]-=h
    fd=(loss_and_grad_r(kern,q,Cs,rp)[0]-loss_and_grad_r(kern,q,Cs,rm)[0])/(2*h)
    print('dL/dr',n,k,g[n,k],fd)
# 2) params through flow
src=Source([7,0,-.5],[[70,0,12],[0,27,0],[12,0,30]],n_coupling=4,hidden=8,seed=3)
for c in src.couplings: c.W2=rng.normal(0,.3,c.W2.shape); c.b2=rng.normal(0,.1,c.b2.shape)
z=rng.standard_normal((200,3))
def lossf():
    r,_=src.sample(z); return loss_and_grad_r(kern,q,Cs,r)[0]
r,ca=src.sample(z); L,gr=loss_and_grad_r(kern,q,Cs,r); grads=src.backward(ca,gr)
worst=0
for mi,(m,gm) in enumerate(zip(src.params_flat(),grads)):
    for k,arr in m.items():
        idx=tuple(rng.integers(0,s) for s in arr.shape)
        h=1e-5; old=arr[idx]; arr[idx]=old+h; lp=lossf(); arr[idx]=old-h; lm=lossf(); arr[idx]=old
        fd=(lp-lm)/(2*h); an=gm[k][idx]; rel=abs(fd-an)/(abs(fd)+abs(an)+1e-12)
        worst=max(worst,rel); print(f'module {mi} {k}{idx}: analytic {an:.6e} fd {fd:.6e} rel {rel:.1e}')
print('worst rel err',worst)
# timing
r=rng.normal(size=(1000,3))*[8,5,5]; q8=qs[np.where(np.linalg.norm(qs,axis=1)>0)[0][::8]]
t=time.time(); loss_and_grad_r(kern,q8,C[np.where(np.linalg.norm(qs,axis=1)>0)[0][::8]],r); print('step time',time.time()-t, len(q8))
# 3) SAS flow
src=SASSource([7,0,-.5],[[70,0,12],[0,27,0],[12,0,30]],n_sas=2)
for m in src.layers:
    if isinstance(m,SAS): m.p_d[:]=rng.normal(0,.2,3); m.e[:]=rng.normal(0,.3,3)
    else: m.p_low[:]=rng.normal(0,.2,3); m.mu[:]=rng.normal(0,.3,3)
z=rng.standard_normal((200,3))
r,ca=src.sample(z); L,gr=loss_and_grad_r(kern,q,Cs,r); grads=src.backward(ca,gr)
worst=0
for mi,(m,gm) in enumerate(zip(src.params_flat(),grads)):
    for k,arr in m.items():
        idx=tuple(rng.integers(0,s) for s in arr.shape)
        h=1e-5; old=arr[idx]; arr[idx]=old+h; lp=lossf(); arr[idx]=old-h; lm=lossf(); arr[idx]=old
        fd=(lp-lm)/(2*h); an=gm[k][idx]; rel=abs(fd-an)/(abs(fd)+abs(an)+1e-12); worst=max(worst,rel)
print('SAS flow worst rel err',worst)
