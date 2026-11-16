# fitting

Fits a trainable normalizing flow (TensorFlow Probability) to a measured/model
three-dimensional correlation function by minimizing a KL-type loss against a tabulated
target `C(q)`, using a differentiable regular-grid linear interpolator.

## Contents

| File | Purpose |
|---|---|
| `flow-optimizer.ipynb` | TensorFlow/TFP version: interpolator, Gaussian (Cholesky), RealNVP and **Levy (alpha-stable, trainable alpha and beta per axis)** sources, loss, Adam training loop, comparison with the true source |
| `65.dat` | Target correlation values, loaded as `true_Cq` and evaluated on a 20x20x20 q-grid (`q_max = 50`) |
| `ppi-<x>.dat` | Large tabulated data files, one per setting x = 0.54 ... 0.79 (~37 MB each) |
| `Kl-ppiplus.dat` | Additional tabulated data file (~8 MB) |

## Requirements

Python 3, `tensorflow`, `tensorflow_probability`, `numpy`.

## Notes

- The seven `ppi-*.dat` files are ~260 MB in total. If the repository gets unwieldy,
  move them to Git LFS or a release asset. `Archive.zip` (a duplicate of those files) is ignored.

Further documentation: [docs/index.md](docs/index.md)

## Running `flow-optimizer.ipynb` (TensorFlow notebook)

The notebook needs TensorFlow **and** TensorFlow Probability, and TFP only works with the legacy Keras 2 package:

```bash
python3.11 -m venv tfenv && source tfenv/bin/activate          # a fresh venv; 3.11 or 3.12 are the safest Python versions for TF
pip install tensorflow tensorflow-probability tf-keras numpy matplotlib jupyter ipykernel
python -c "import tensorflow as tf, tensorflow_probability as tfp; print(tf.__version__, tfp.__version__)"   # run this in Terminal first
python -m ipykernel install --user --name tfenv && jupyter lab   # pick the `tfenv` kernel
```

If the kernel dies on `import tensorflow`, run that one-liner in Terminal: the traceback or abort message shows the real cause
(a notebook kernel hides it). Typical causes: a Python/TF version mismatch, an x86_64 Python running under Rosetta on an Apple-silicon Mac
(check `python -c "import platform; print(platform.machine())"` prints `arm64`), a second OpenMP runtime (`OMP: Error #15`; last-resort workaround
`KMP_DUPLICATE_LIB_OK=TRUE`, commented out in the first notebook cell), or Keras 3 without `tf-keras` (TFP then fails to import).
The NumPy code (`srcfit.py`, `train.py`) does not need any of this.
