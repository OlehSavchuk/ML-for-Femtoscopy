# fitting

Fits a trainable normalizing flow (TensorFlow Probability) to a measured/model
three-dimensional correlation function by minimizing a KL-type loss against a tabulated
target `C(q)`, using a differentiable regular-grid linear interpolator.

## Contents

| File | Purpose |
|---|---|
| `flow-optimizer.ipynb` | Interpolator, trainable Cholesky/affine bijector flow, loss, Adam training loop |
| `65.dat` | Target correlation values, loaded as `true_Cq` and evaluated on a 20x20x20 q-grid (`q_max = 50`) |
| `ppi-<x>.dat` | Large tabulated data files, one per setting x = 0.54 ... 0.79 (~37 MB each) |
| `Kl-ppiplus.dat` | Additional tabulated data file (~8 MB) |

## Requirements

Python 3, `tensorflow`, `tensorflow_probability`, `numpy`.

## Notes

- The seven `ppi-*.dat` files are ~260 MB in total. If the repository gets unwieldy,
  move them to Git LFS or a release asset. `Archive.zip` (a duplicate of those files) is ignored.

Further documentation: [docs/index.md](docs/index.md)
