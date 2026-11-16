"""Compare fitted sources (Gaussian / sinh-arcsinh flow, Gaussian or Levy-stable base) with the known source ppi-0.65.
usage: python3 evaluate_levy.py   (reads results/fit_*.pkl, writes results/fig_levy_*.png and results/metrics_levy.json)"""
import pickle, json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from srcfit import *

D = os.environ.get('FIT_DATA', '.') + '/'
kern = Kernel(D + 'Kl-ppiplus.dat'); qs = make_qgrid()
nz = np.where(np.linalg.norm(qs, axis=1) > 0)[0]
C = {'real': np.loadtxt(D + '65.dat'), 'syn': np.load('results/C_syn.npy')}
NS = 10000
ref = np.loadtxt(D + 'ppi-0.65.dat')[:NS]
plt.rcParams.update({'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.minor.visible': True, 'ytick.minor.visible': True, 'font.size': 10})
runs = {  # label -> (pickle, target)
    'Gaussian base, Gaussian': ('gauss', 'real'), 'Levy 1.5 base, Gaussian': ('gauss_levy1.5', 'real'),
    'Gaussian base, SAS flow': ('sas', 'real'), 'Levy 1.5 base, SAS flow': ('sas_levy1.5', 'real'),
    'closed test: Gaussian, Gaussian base': ('gauss_syn', 'syn'), 'closed test: SAS flow, Gaussian base': ('sas_syn', 'syn'),
    'closed test: SAS flow, Levy 1.9 base': ('sas_syn_levy1.9', 'syn'), 'closed test: SAS flow, Levy 1.5 base': ('sas_syn_levy1.5', 'syn'),
}
def w1(a, b): return float(np.mean(np.abs(np.sort(a) - np.sort(b))))
def stats(r):
    z = (r - r.mean(0)) / r.std(0)
    return dict(median=np.median(r, 0).round(2).tolist(), iqr=(np.percentile(r, 75, 0) - np.percentile(r, 25, 0)).round(2).tolist(),
                skew=(z ** 3).mean(0).round(2).tolist(), p99=np.percentile(np.abs(r - np.median(r, 0)), 99, 0).round(1).tolist())
rows, models = [], {}
for lab, (f, tg) in runs.items():
    p = f'results/fit_{f}.pkl'
    if not os.path.exists(p): continue
    src, hist = pickle.load(open(p, 'rb'))
    r = sample_source(src, NS); models[lab] = (r, tg)
    c = correlation(kern, qs[nz], r, chunk=20)
    rows.append(dict(name=lab, target=tg, C_rms=float(np.sqrt(np.mean((c - C[tg][nz]) ** 2))), **stats(r),
                     W1=[round(w1(r[:, i], ref[:, i]), 2) for i in range(3)],
                     W1_r=round(w1(np.linalg.norm(r, axis=1), np.linalg.norm(ref, axis=1)), 2)))
c = correlation(kern, qs[nz], ref, chunk=20)
rows.append(dict(name='ppi-0.65 (true)', target='-', C_rms_to_65=float(np.sqrt(np.mean((c - C['real'][nz]) ** 2))),
                 C_rms_to_syn=float(np.sqrt(np.mean((c - C['syn'][nz]) ** 2))), **stats(ref)))
json.dump(rows, open('results/metrics_levy.json', 'w'), indent=1)
for x in rows: print(x)

col = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:cyan', 'tab:olive']
for tg, fname in [('real', 'fig_levy_real.png'), ('syn', 'fig_levy_closed.png')]:
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for k, (lab, (r, t)) in enumerate(models.items()):
        if t != tg: continue
        for i in range(3):
            b = np.linspace(-45, 60, 120) if i == 0 else np.linspace(-45, 45, 120)
            ax[i].hist(np.clip(r[:, i], b[0], b[-1]), bins=b, density=True, histtype='step', lw=1.6, color=col[k], label=lab)
        ax[3].hist(np.clip(np.linalg.norm(r, axis=1), 0, 80), bins=np.linspace(0, 80, 100), density=True, histtype='step', lw=1.6, color=col[k])
    for i in range(3):
        b = np.linspace(-45, 60, 120) if i == 0 else np.linspace(-45, 45, 120)
        ax[i].hist(np.clip(ref[:, i], b[0], b[-1]), bins=b, density=True, histtype='step', lw=2.4, color='k', label='ppi-0.65 (true)')
        ax[i].set_yscale('log'); ax[i].set_ylim(1e-5, None); ax[i].set_xlabel(['x (out)', 'y (side)', 'z (long)'][i] + ' [fm]'); ax[i].set_ylabel('density')
    ax[3].hist(np.clip(np.linalg.norm(ref, axis=1), 0, 80), bins=np.linspace(0, 80, 100), density=True, histtype='step', lw=2.4, color='k')
    ax[3].set_xlabel('|r| [fm]'); ax[3].set_yscale('log'); ax[3].set_ylim(1e-5, None)
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle('target: ' + ('65.dat (measured C)' if tg == 'real' else 'C computed from ppi-0.65 (closed test)') + ';  tails clipped at the axis limits', fontsize=10)
    fig.tight_layout(); fig.savefig('results/' + fname, dpi=130)
