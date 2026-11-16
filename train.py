"""usage: python3 train.py <gauss|realnvp|sas> <steps> [real|syn] [alpha=2]
alpha < 2: latent distribution is symmetric Levy-stable with that index (2 = Gaussian, 1 = Cauchy)
real: target is 65.dat (the measured correlation function)
syn : closed test, target is C computed from ppi-0.65 with the same kernel (the answer is known and exactly representable)"""
import sys, pickle, time, os, numpy as np
from srcfit import *
D = os.environ.get('FIT_DATA', '.') + '/'; os.makedirs('results', exist_ok=True)
model, steps = sys.argv[1], int(sys.argv[2])
target = sys.argv[3] if len(sys.argv) > 3 else 'real'
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
kern = Kernel(D + 'Kl-ppiplus.dat'); qs = make_qgrid()
nz = np.where(np.linalg.norm(qs, axis=1) > 0)[0]; sel = nz[::8]          # same stride as the notebook, q=0 excluded
if target == 'real':
    C = np.loadtxt(D + '65.dat')
else:
    f = 'results/C_syn.npy'
    if not os.path.exists(f):
        np.save(f, correlation(kern, qs, np.loadtxt(D + 'ppi-0.65.dat')[:40000], chunk=20))
    C = np.load(f)
q, Cs = qs[sel], C[sel]
mean0 = [4.0, 0.0, -0.5]; cov0 = np.diag([80., 30., 30.])               # notebook's initial guess
if model == 'sas':
    src = SASSource(mean0, cov0, n_sas=2)
else:
    src = Source(mean0, cov0, n_coupling=0 if model == 'gauss' else 6, hidden=32, seed=0)
src.alpha = alpha
lr = {'gauss': 5e-3, 'realnvp': 2e-3, 'sas': 3e-3}[model]
t = time.time()
hist = fit(kern, q, Cs, src, steps, n_samples=1000, lr=lr, log_every=200,
           log=lambda s: print(s, f'[{time.time()-t:.0f}s]', flush=True))
tag = (model if target == 'real' else model + '_syn') + ('' if alpha >= 2 else '_levy%g' % alpha)
pickle.dump((src, hist), open(f'results/fit_{tag}.pkl', 'wb'))
print('done', time.time() - t)
