import sys, pickle, time, numpy as np
from srcfit import *
import os
D=os.environ.get('FIT_DATA','.')+'/'; os.makedirs('results',exist_ok=True)
model, steps = sys.argv[1], int(sys.argv[2])
kern=Kernel(D+'Kl-ppiplus.dat'); C=np.loadtxt(D+'65.dat'); qs=make_qgrid()
nz=np.where(np.linalg.norm(qs,axis=1)>0)[0]; sel=nz[::8]          # same stride as the notebook, q=0 excluded
q,Cs=qs[sel],C[sel]
mean0=[4.0,0.0,-0.5]; cov0=np.diag([80.,30.,30.])               # notebook's initial guess
src = Source(mean0,cov0,n_coupling=0 if model=='gauss' else 6,hidden=32,seed=0)
t=time.time()
hist=fit(kern,q,Cs,src,steps,n_samples=1000,lr=5e-3 if model=='gauss' else 2e-3,log_every=100,
         log=lambda s:print(s,f'[{time.time()-t:.0f}s]',flush=True))
pickle.dump((src,hist),open(f'results/fit_{model}.pkl','wb'))
print('done',time.time()-t)
