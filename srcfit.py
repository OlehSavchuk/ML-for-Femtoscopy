"""
srcfit.py - fit a 3D femtoscopic source S(r) to a measured correlation function C(q).

Model:  C(q_j) ~ (1/N) sum_n K(|q_j|, |r_n|, angle(q_j, r_n)),   r_n ~ S
K is the tabulated kernel (Kl-ppiplus.dat) on a regular (|q|, |r|, theta) grid.

Parametrisations of S (all with analytic gradients, pure numpy):
  * Source(n_coupling=0) : Gaussian  r = L z + mu   (z ~ N(0, I) or Levy-stable, L lower-triangular, positive diagonal)
  * Source(n_coupling=k) : RealNVP   r = A(coupling_k(...coupling_1(z)))   (affine couplings, then the affine map)
  * SASSource            : short flow of sinh-arcsinh bijectors between full linear maps (skewness + tail weight)
The latent z is Gaussian by default; set src.alpha < 2 for a Levy-stable base (sample_base).
All start exactly at the Gaussian model (zero-initialised couplings, identity SAS).

This mirrors the TensorFlow notebook (flow-optimizer.ipynb) but runs without TensorFlow and
is what produced the numbers in results/.
"""
import numpy as np

XMAX = np.array([100.0, 40.0, np.pi])   # kernel axis ranges: |q|, |r|, theta
NG = 100


# ----------------------------------------------------------------------------- kernel
class Kernel:
    def __init__(self, path):
        self.K = np.loadtxt(path, dtype=np.float64).reshape(NG, NG, NG)
        self.Kf = self.K.ravel()

    def __call__(self, q, rho, th, grad=False):
        """Trilinear interpolation with constant extension outside the grid (as tfp does).
        q, rho, th broadcast to the same shape. If grad, also returns dK/drho and dK/dtheta."""
        u = np.stack([q / XMAX[0], rho / XMAX[1], th / XMAX[2]]) * (NG - 1)
        inside = (u >= 0) & (u <= NG - 1)
        uc = np.clip(u, 0, NG - 1)
        i0 = np.minimum(np.floor(uc).astype(np.int64), NG - 2)
        f = uc - i0
        a, b, c = i0
        fa, fb, fc = f
        s = NG * NG, NG, 1
        base = a * s[0] + b * s[1] + c
        K = self.Kf
        k000, k001 = K[base], K[base + 1]
        k010, k011 = K[base + s[1]], K[base + s[1] + 1]
        k100, k101 = K[base + s[0]], K[base + s[0] + 1]
        k110, k111 = K[base + s[0] + s[1]], K[base + s[0] + s[1] + 1]
        c00 = k000 * (1 - fc) + k001 * fc
        c01 = k010 * (1 - fc) + k011 * fc
        c10 = k100 * (1 - fc) + k101 * fc
        c11 = k110 * (1 - fc) + k111 * fc
        c0 = c00 * (1 - fb) + c01 * fb
        c1 = c10 * (1 - fb) + c11 * fb
        val = c0 * (1 - fa) + c1 * fa
        if not grad:
            return val
        d_b = ((c01 - c00) * (1 - fa) + (c11 - c10) * fa) * (NG - 1) / XMAX[1] * inside[1]
        dc0 = (k001 - k000) * (1 - fb) + (k011 - k010) * fb
        dc1 = (k101 - k100) * (1 - fb) + (k111 - k110) * fb
        d_c = (dc0 * (1 - fa) + dc1 * fa) * (NG - 1) / XMAX[2] * inside[2]
        return val, d_b, d_c


def make_qgrid(n=20, qmax=50.0):
    """q grid in the same order as the notebook loop (index = k + n*j + n*n*i)."""
    g = np.arange(n) * (2 * qmax / n) - qmax
    I, J, Kk = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([I.ravel(), J.ravel(), Kk.ravel()], 1)


EPS_SIN = 1e-3     # keeps d(acos) finite near theta = 0, pi (the NaN source in the original loss)


def correlation(kern, qs, r, chunk=250):
    """C(q) = mean_n K(|q|, |r_n|, theta_qn).  qs: (M,3), r: (N,3). Returns (M,)"""
    rho = np.linalg.norm(r, axis=1)[None, :] + 1e-12
    out = []
    for i in range(0, len(qs), chunk):
        qc = qs[i:i + chunk]
        qn = np.linalg.norm(qc, axis=1)[:, None] + 1e-12
        cos = np.clip(qc @ r.T / (qn * rho), -1, 1)
        th = np.arccos(cos)
        out.append(kern(np.broadcast_to(qn, th.shape), np.broadcast_to(rho, th.shape), th).mean(1))
    return np.concatenate(out)


def loss_and_grad_r(kern, qs, C, r):
    """L = || mean_n K - C ||_2   and  dL/dr  (N,3)."""
    M, N = len(qs), len(r)
    rho = np.linalg.norm(r, axis=1) + 1e-12
    qn = np.linalg.norm(qs, axis=1) + 1e-12
    qh = qs / qn[:, None]
    cos = np.clip(qh @ r.T / rho[None, :], -1, 1)
    th = np.arccos(cos)
    val, dKr, dKt = kern(np.broadcast_to(qn[:, None], th.shape),
                         np.broadcast_to(rho[None, :], th.shape), th, grad=True)
    m = val.mean(1)
    resid = m - C
    L = np.linalg.norm(resid) + 1e-30
    A = (resid / L)[:, None] / N                      # dL/dK_mn
    g_rho = (A * dKr).sum(0)                          # (N,)
    sin = np.maximum(np.sqrt(1 - cos ** 2), EPS_SIN)
    W = A * dKt / sin                                 # dtheta = -dcos/sin
    # dcos/dr_n = qh/rho - cos * r/rho^2
    g_r = g_rho[:, None] * r / rho[:, None]
    g_r += -(W.T @ qh) / rho[:, None] + (r / rho[:, None] ** 2) * (W * cos).sum(0)[:, None]
    return L, g_r


# ----------------------------------------------------------------------------- models
def sample_base(rng, n, alpha=2.0, dim=3):
    """Latent distribution: i.i.d. symmetric alpha-stable (Levy-stable) components, Chambers-Mallows-Stuck sampling,
    scale chosen so that alpha = 2 is exactly the standard normal. alpha < 2 gives power-law tails (infinite variance),
    alpha = 1 is a Cauchy distribution. The fitted source is then  r = flow(z), z ~ Levy-stable."""
    if alpha >= 2.0:
        return rng.standard_normal((n, dim))
    V = rng.uniform(-np.pi / 2, np.pi / 2, (n, dim))
    W = rng.exponential(1.0, (n, dim))
    if abs(alpha - 1.0) < 1e-9:
        x = np.tan(V)
    else:
        x = (np.sin(alpha * V) / np.cos(V) ** (1 / alpha)) * (np.cos((1 - alpha) * V) / W) ** ((1 - alpha) / alpha)
    return x / np.sqrt(2.0)


def softplus(x):
    return np.logaddexp(0, x)


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


class Affine:
    """r = z L^T + mu, L lower-triangular, diagonal = softplus(p) + 1e-3 (always positive!)."""
    def __init__(self, mean, cov):
        L = np.linalg.cholesky(cov)
        d = np.diag(L) - 1e-3
        self.p_diag = np.log(np.expm1(d))             # inverse softplus
        self.p_low = np.array([L[1, 0], L[2, 0], L[2, 1]])
        self.mu = np.array(mean, float)

    def params(self):
        return {"p_diag": self.p_diag, "p_low": self.p_low, "mu": self.mu}

    def L(self):
        L = np.zeros((3, 3))
        L[np.diag_indices(3)] = softplus(self.p_diag) + 1e-3
        L[1, 0], L[2, 0], L[2, 1] = self.p_low
        return L

    def forward(self, z):
        L = self.L()
        return z @ L.T + self.mu, (z, L)

    def backward(self, cache, dr):
        z, L = cache
        dL = dr.T @ z
        g = {"p_diag": np.diag(dL) * sigmoid(self.p_diag),
             "p_low": np.array([dL[1, 0], dL[2, 0], dL[2, 1]]),
             "mu": dr.sum(0)}
        return g, dr @ L


class Coupling:
    """Affine coupling: y_b = x_b * exp(S tanh(s)) + t,  (s, t) = MLP(x_a)."""
    def __init__(self, cond_idx, hidden, rng, S=1.5):
        self.a = list(cond_idx)
        self.b = [i for i in range(3) if i not in self.a]
        self.S = S
        da, db = len(self.a), len(self.b)
        self.W1 = rng.normal(0, 1 / np.sqrt(da), (da, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = np.zeros((hidden, 2 * db))          # zero init => identity map
        self.b2 = np.zeros(2 * db)

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        xa, xb = x[:, self.a], x[:, self.b]
        h = np.tanh(xa @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        db = len(self.b)
        s = self.S * np.tanh(o[:, :db])
        t = o[:, db:]
        y = x.copy()
        y[:, self.b] = xb * np.exp(s) + t
        return y, (xa, xb, h, s)

    def backward(self, cache, dy):
        xa, xb, h, s = cache
        dya, dyb = dy[:, self.a], dy[:, self.b]
        es = np.exp(s)
        dxb = dyb * es
        ds_raw = dyb * xb * es * self.S * (1 - (s / self.S) ** 2)
        do = np.concatenate([ds_raw, dyb], 1)
        g = {"W2": h.T @ do, "b2": do.sum(0)}
        dpre = (do @ self.W2.T) * (1 - h ** 2)
        g["W1"], g["b1"] = xa.T @ dpre, dpre.sum(0)
        dx = np.zeros_like(dy)
        dx[:, self.a] = dya + dpre @ self.W1.T
        dx[:, self.b] = dxb
        return g, dx


class Source:
    """Flow  z -> r. layers = coupling layers (may be empty = pure Gaussian) followed by an Affine."""
    def __init__(self, mean, cov, n_coupling=0, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        masks = [[0], [1, 2], [2], [0, 1], [1], [0, 2]]
        self.couplings = [Coupling(masks[i % len(masks)], hidden, rng) for i in range(n_coupling)]
        self.affine = Affine(mean, cov)

    def modules(self):
        return self.couplings + [self.affine]

    def sample(self, z):
        caches = []
        x = z
        for m in self.modules():
            x, c = m.forward(x)
            caches.append(c)
        return x, caches

    def backward(self, caches, dr):
        grads = [None] * len(caches)
        d = dr
        for i in reversed(range(len(caches))):
            grads[i], d = self.modules()[i].backward(caches[i], d)
        return grads

    def params_flat(self):
        return [m.params() for m in self.modules()]


class SAS:
    """Element-wise sinh-arcsinh bijector (Jones & Pewsey): y = sinh(d * asinh(x) + e), d = exp(p_d) > 0.
    e controls the skewness, d the tail weight (d > 1 heavier tails, d < 1 lighter); d = 1, e = 0 is the identity.
    It maps a Gaussian onto skewed / heavy-tailed marginals with only 2 parameters per axis."""
    VMAX = 12.0

    def __init__(self, dim=3):
        self.p_d = np.zeros(dim)
        self.e = np.zeros(dim)

    def params(self):
        return {"p_d": self.p_d, "e": self.e}

    def forward(self, x):
        d = np.exp(self.p_d)
        u = np.arcsinh(x)
        v = np.clip(d * u + self.e, -self.VMAX, self.VMAX)
        return np.sinh(v), (x, u, v, d)

    def backward(self, cache, dy):
        x, u, v, d = cache
        ch = np.cosh(v)
        live = (np.abs(d * u + self.e) < self.VMAX)
        g = {"p_d": (dy * ch * u * d * live).sum(0), "e": (dy * ch * live).sum(0)}
        return g, dy * ch * d * live / np.sqrt(1 + x * x)


class SASSource(Source):
    """Short flow with non-Gaussian bijectors:  z -> Affine -> SAS -> Affine -> SAS -> Affine(mean, cov0).
    ~40 parameters; starts exactly at the Gaussian model (SAS = identity, inner affines = identity)."""
    def __init__(self, mean, cov, n_sas=2, seed=0):
        self.couplings = []
        self.affine = Affine(mean, cov)
        self.layers = []
        for _ in range(n_sas):
            self.layers += [Affine([0, 0, 0], np.eye(3)), SAS()]

    def modules(self):
        return self.layers + [self.affine]


class Adam:
    def __init__(self, src, lr=3e-3, b1=0.9, b2=0.999):
        self.src, self.lr, self.b1, self.b2, self.t = src, lr, b1, b2, 0
        self.m = [{k: np.zeros_like(v) for k, v in p.items()} for p in src.params_flat()]
        self.v = [{k: np.zeros_like(v) for k, v in p.items()} for p in src.params_flat()]

    def step(self, grads, lr=None):
        lr = lr or self.lr
        self.t += 1
        for params, g, m, v in zip(self.src.params_flat(), grads, self.m, self.v):
            for k in params:
                m[k] = self.b1 * m[k] + (1 - self.b1) * g[k]
                v[k] = self.b2 * v[k] + (1 - self.b2) * g[k] ** 2
                mh = m[k] / (1 - self.b1 ** self.t)
                vh = v[k] / (1 - self.b2 ** self.t)
                params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)     # in-place on the module arrays


def fit(kern, qs, C, src, steps, n_samples=1000, lr=3e-3, seed=1, log_every=100, cosine=True, log=print,
        train_modules=None):
    rng = np.random.default_rng(seed)
    opt = Adam(src, lr=lr)
    hist = []
    for it in range(steps):
        z = sample_base(rng, n_samples, getattr(src, 'alpha', 2.0))
        r, caches = src.sample(z)
        L, g_r = loss_and_grad_r(kern, qs, C, r)
        grads = src.backward(caches, g_r)
        if train_modules is not None:
            for i, m in enumerate(src.modules()):
                if m not in train_modules:
                    grads[i] = {k: np.zeros_like(v) for k, v in grads[i].items()}
        cur = lr * (0.5 * (1 + np.cos(np.pi * it / steps)) * 0.97 + 0.03) if cosine else lr
        opt.step(grads, cur)
        hist.append(L)
        if log_every and (it % log_every == 0 or it == steps - 1):
            log(f"  step {it:5d}  loss {np.mean(hist[-log_every:]):.5f}")
    return np.array(hist)


def sample_source(src, n, seed=123):
    z = sample_base(np.random.default_rng(seed), n, getattr(src, 'alpha', 2.0))
    return src.sample(z)[0]
