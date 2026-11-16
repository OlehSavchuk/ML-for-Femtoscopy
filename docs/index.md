# fitting: source reconstruction with normalizing flows

## Problem

Given a kernel `K(|q|, |r|, theta)` (`Kl-ppiplus.dat`, 100x100x100 table, `|q|` in [0,100] MeV/c, `|r|` in [0,40] fm, `theta` in [0,pi]) and a
measured/model correlation function `C(q)` on a 20x20x20 grid `q_i = -50 + 5 i` (`65.dat`, 8000 values, C-order with `q_x` slowest), find a 3D source `S(r)` with

```
C(q) = < K(|q|, |r|, angle(q, r)) >_{r ~ S}
```

The source is represented by a flow `r = f(z)`, `z ~ N(0, I)`, and trained by minimising `|| <K>_r - C ||_2` on 1000 `q` points (every 8th non-zero grid point) with 1000 fresh samples per step.

## Data

* `ppi-<x>.dat`: 500 000 samples `(x, y, z)` [fm] of the source for settings `x = 0.54 ... 0.79`. These are the **real sources**; they are the reference for comparison.
* `65.dat`: target `C(q)`.
* Pushing a `ppi` source through the kernel gives `C` with rms difference to `65.dat` of 0.0049-0.011 (best: `ppi-0.73`/`0.69`, `ppi-0.65` gives 0.0055),
  independent of sample size, so `65.dat` is close to, but not exactly, the correlation function of any of these files with this kernel
  (axis order and signs were scanned: the identity convention is the best one).

## Models

1. **Gaussian**: `r = L z + mu`, `L` lower-triangular with positive (softplus) diagonal. 9 parameters.
2. **RealNVP**: 6 affine coupling layers (tanh MLP with 32 hidden units, bounded log-scale `1.5 tanh`, coordinate rotation between layers) followed by the same affine map. The last MLP layers start at zero, so it starts as the Gaussian.

## Bugs fixed in the original notebook

1. `Cholesky` bijector: comment promised a positive diagonal, but raw entries were used, so `log(diag)` could be `NaN`. Now softplus.
2. `_shift` was declared (initialised at 7, target mean 4) but never used in `forward`/`inverse`; the source mean could not be fitted. Now applied.
3. Loss: `acos` gradient is infinite at `+-1` (clipping the value does not protect the gradient) and `0/0` at `q = 0` or `r = 0`. Now clipped inside `(-1, 1)` with an epsilon.
4. `lr_decay` was unused (now a cosine schedule); duplicate dead `KL_Divergence` removed.

## Code

| File | Role |
|---|---|
| `flow-optimizer.ipynb` | TensorFlow Probability version (Gaussian + RealNVP + comparison). **Not executed** (no TensorFlow where it was written); syntax-checked only. |
| `srcfit.py` | Pure numpy implementation with analytic gradients (kernel interpolation, Gaussian and RealNVP flows, Adam) |
| `check_gradients.py` | Finite-difference check of every gradient (worst relative error 1.6e-5) |
| `train.py <gauss\|realnvp> <steps>` | Training, writes `results/fit_*.pkl` |
| `evaluate.py` | Metrics and figures in `results/` |

Run with `FIT_DATA=. python train.py gauss 1500`, `python train.py realnvp 3000`, `python evaluate.py` (about 5 and 10 minutes on 2 cores).

## Results (numpy implementation, 20 000 samples per source, all 7999 non-zero `q`)

| source | rms(C - C_65) | mean (x, y, z) | std (x, y, z) | cov_xz |
|---|---|---|---|---|
| initial guess (diag(80,30,30), mean (4,0,-0.5)) | 0.00512 | 4.03, 0.01, -0.51 | 8.95, 5.46, 5.48 | -0.3 |
| Gaussian fit | 0.00238 | 5.02, -0.10, -0.87 | 7.98, 6.55, 4.51 | 3.5 |
| RealNVP fit | 0.00117 | 7.30, -0.39, -0.57 | 9.23, 8.55, 4.93 | 9.0 |
| **true source `ppi-0.65`** | 0.00551 | 7.26, -0.01, -0.57 | 8.36, 5.21, 5.38 | 12.8 |

(`results/metrics.json` has all seven true sources, skewness and 1D Wasserstein distances; figures: `results/fig_source_comparison.png`, `fig_correlation_fit.png`, `fig_loss.png`.)

## Interpretation

* Both fits reproduce `C` better than the **true** sources do (0.0024 and 0.0012 vs 0.0055). The fit therefore absorbs part of the mismatch between `65.dat` and the kernel-model, and the fitted source is **not** the true one.
* The RealNVP recovers the mean in `x` (7.3 vs 7.26) and the `x` tail, but is too wide in `y` (8.6 vs 5.2) and too narrow/peaked in `z`.
* `C(q)` on this grid does not determine `S(r)` uniquely (3D source from a 3D correlation through an angular kernel with noise). Useful next steps: fit to `C` computed from a `ppi` file (a closed test with a known answer), add a prior/regulariser (e.g. cut at `|y| < 21 fm` seen in the data, or a penalty on `|r| > 40` where the kernel table ends), and compare several `q` subsets.
