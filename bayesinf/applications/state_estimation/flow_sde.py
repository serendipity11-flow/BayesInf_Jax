"""
RealNVPFlowSDE: Flow-SDE for posterior trajectory estimation.

Model:
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(eps_t, z_t, c_t, t) \sqrt(dt),
    eps_t ~ N(0, I) sampled independently per solver step.

    c_t = BiGRUEncoder(y_{1:T})_t  — per-step context from a bidirectional GRU
          over the entire observation sequence (smoothing: c_t sees future y's).

    f_θ — neural drift (MLP). To add a physics prior, wrap externally:
          f_total = physics_fn(z_t, t) + model.drift(z_t, c_t, t).

    G_φ — conditional RealNVP transition-noise transform. Each affine coupling
          uses log_scale = alpha * tanh(raw_scale), bounding every diagonal
          scale to the open interval (exp(-alpha), exp(alpha)).

Training:
    Exact change-of-variables transition NLL. The custom solver path uses
    stochastic segment starts z0 ~ N(mu_psi(y_segment), diag(sigma_psi^2))
    from a GRU inference network, plus optional adjacent-segment continuity loss.

Inference — Euler-Maruyama (or diffrax Heun) particle simulation conditioned
on the BiGRU encoding of y_{1:T}.
"""

import os
import numpy as np
from numba import cuda

if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.default_device("cpu")

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def energy_score(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    beta: float = 1.0,
    uncertainty_weight: float = 0.5,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Multivariate CRPS for an ensemble of paths with adjustable uncertainty control.

    Args:
        preds: Tensor of shape (M, T, d_x) representing M simulated ensemble paths.
        target: Tensor of shape (T, d_x) representing the true path.
        beta: Power exponent, must be strictly in the range (0, 2).
        uncertainty_weight: Weight gamma for the pairwise spread term. 
                            Set > 0.5 to force wider prediction intervals (higher coverage).
        eps: Small stability constant to prevent NaN gradients near zero distance.
    """
    # 1. Reconstruction Loss: Distance of ensemble members to target
    diff_target = preds - target[None, :, :]
    dist_target = jnp.sqrt(jnp.sum(diff_target**2, axis=-1) + eps)
    if beta != 1.0:
        dist_target = dist_target ** beta
    mean_dist_target = jnp.mean(dist_target, axis=0)

    # 2. Uncertainty Control: Distance between ensemble members
    diff_pairwise = preds[:, None, :, :] - preds[None, :, :, :]
    dist_pairwise = jnp.sqrt(jnp.sum(diff_pairwise**2, axis=-1) + eps)
    if beta != 1.0:
        dist_pairwise = dist_pairwise ** beta
    mean_dist_pairwise = jnp.mean(dist_pairwise, axis=(0, 1))

    # Balanced loss combination
    return jnp.mean(mean_dist_target - uncertainty_weight * mean_dist_pairwise)


def gaussian_nll(preds: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Diagonal Gaussian NLL fitted to an ensemble of paths."""
    mu = jnp.mean(preds, axis=0)
    var = jnp.var(preds, axis=0) + eps
    nll = 0.5 * ((target - mu) ** 2) / var + 0.5 * jnp.log(2.0 * jnp.pi * var)
    return jnp.mean(nll)


def _positive_dt(t0: jnp.ndarray, t1: jnp.ndarray) -> jnp.ndarray:
    """Return a strictly positive step size for stable sqrt(dt)."""
    return jnp.maximum(t1 - t0, 1e-12)


# Import custom stochastic solvers
from bayesinf.applications.state_estimation.stochastic_solvers import (
    EulerMaruyama,
    StochasticHeun,
    ReversibleHeun,
    Milstein,
    StochasticMidpoint,
    AdaptiveEulerMaruyama,
    StochasticRK4,
    StepNoisePath,
    VirtualBrownianTree,
    get_solver as get_stochastic_solver,
    list_solvers,
)


class SinusoidalEmb(eqx.Module):
    """Sinusoidal positional embedding for a scalar time value -> (dim,)."""
    dim: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        half  = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / float(max(1, half - 1)))
        args  = jnp.asarray(t, dtype=jnp.float32) * freqs
        emb   = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros(1)])
        return emb


# ---------------------------------------------------------------------------
# Bidirectional GRU encoder  y_{1:T} -> c_{1:T}
# ---------------------------------------------------------------------------

class BiGRUEncoder(eqx.Module):
    """
    Bidirectional GRU encoder producing per-step context vectors.

    Forward and backward hidden states are concatenated then projected.
    This lets c_t attend to both past and future observations (smoothing).

    Input : y  (T, d_y)
    Output: c  (T, d_out)
    """
    fwd_cell: eqx.nn.GRUCell
    bwd_cell: eqx.nn.GRUCell
    proj:     eqx.nn.Linear

    def __init__(self, d_y: int, d_hidden: int, d_out: int, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.fwd_cell = eqx.nn.GRUCell(d_y, d_hidden, key=k1)
        self.bwd_cell = eqx.nn.GRUCell(d_y, d_hidden, key=k2)
        self.proj     = eqx.nn.Linear(2 * d_hidden, d_out, key=k3)

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        d_h = self.fwd_cell.hidden_size
        h0  = jnp.zeros(d_h)

        def fwd_step(h, y_t):
            h_new = self.fwd_cell(y_t, h)
            return h_new, h_new

        def bwd_step(h, y_t):
            h_new = self.bwd_cell(y_t, h)
            return h_new, h_new

        _, h_fwd = jax.lax.scan(fwd_step, h0, y)           # (T, d_h)
        _, h_bwd = jax.lax.scan(bwd_step, h0, y[::-1])     # (T, d_h), reversed
        h_bwd    = h_bwd[::-1]                              # restore time order

        h_bi = jnp.concatenate([h_fwd, h_bwd], axis=-1)    # (T, 2*d_h)
        return jax.vmap(self.proj)(h_bi)                    # (T, d_out)


class GRUInitialInference(eqx.Module):
    """
    Diagonal-Gaussian segment encoder h_psi(y_segment) -> (mu_0, sigma_0).

    A unidirectional GRU reads one observation segment and parameterises the
    stochastic initial state used to start the RealNVP SDE integration.
    """
    cell:           eqx.nn.GRUCell
    mu_head:        eqx.nn.Linear
    raw_sigma_head: eqx.nn.Linear
    d_hidden:       int = eqx.field(static=True)
    min_sigma:      float = eqx.field(static=True)

    def __init__(
        self,
        d_y: int,
        d_hidden: int,
        d_x: int,
        key: jr.PRNGKey,
        min_sigma: float = 1e-3,
    ):
        k_cell, k_mu, k_sigma = jr.split(key, 3)
        self.cell = eqx.nn.GRUCell(d_y, d_hidden, key=k_cell)
        self.mu_head = eqx.nn.Linear(d_hidden, d_x, key=k_mu)
        self.raw_sigma_head = eqx.nn.Linear(d_hidden, d_x, key=k_sigma)
        self.d_hidden = d_hidden
        self.min_sigma = min_sigma

    def __call__(self, y_seg: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h0 = jnp.zeros(self.d_hidden)

        def step(h, y_t):
            h_new = self.cell(y_t, h)
            return h_new, h_new

        h_last, _ = jax.lax.scan(step, h0, y_seg)
        mu = self.mu_head(h_last)
        sigma = jax.nn.softplus(self.raw_sigma_head(h_last)) + self.min_sigma
        return mu, sigma


# ---------------------------------------------------------------------------
# Drift network  f_θ(z_t, c_t, t)
# ---------------------------------------------------------------------------

class SDEDriftNet(eqx.Module):
    """
    Parameterises the SDE drift  f_θ(z_t, c_t, t) → (d_x,).
    Input:  z_t  (d_x,)  |  c_t  (d_c,)  |  t  scalar physical time
    """
    net:      eqx.nn.MLP
    time_emb: SinusoidalEmb

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        depth:        int,
        time_emb_dim: int,
        key:          jr.PRNGKey,
        output_dim:    int | None = None,
    ):
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size    = d_x + d_c + time_emb_dim,
            out_size   = d_x if output_dim is None else output_dim,
            width_size = d_hidden,
            depth      = depth,
            activation = jax.nn.gelu,
            key        = key,
        )

    def __call__(self, z: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t_emb = self.time_emb(t)
        return self.net(jnp.concatenate([z, c, t_emb]))


# ---------------------------------------------------------------------------
# RealNVP transition-noise network  T_φ(eps_t; z_t, c_t, t)
# ---------------------------------------------------------------------------

class RealNVPDiffusionLayer(eqx.Module):
    """
    Conditional affine coupling layer. The transformed half is
    y_b = x_b * exp(alpha * tanh(s_raw)) + shift.
    """
    net:     eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x:     int  = eqx.field(static=True)
    d_half:  int  = eqx.field(static=True)
    d_pass:  int  = eqx.field(static=True)
    d_shift: int  = eqx.field(static=True)
    flip:    bool = eqx.field(static=True)
    alpha:   float = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        time_emb_dim: int,
        flip:         bool,
        alpha:        float,
        key:          jr.PRNGKey,
    ):
        self.d_x    = d_x
        self.d_half = d_x // 2
        self.flip   = flip
        self.alpha  = alpha
        if flip:
            self.d_pass  = d_x - d_x // 2   # lower half of state
            self.d_shift = d_x // 2          # upper half
        else:
            self.d_pass  = d_x // 2          # upper half of state
            self.d_shift = d_x - d_x // 2    # lower half
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size    = self.d_pass + d_x + d_c + time_emb_dim,
            out_size   = 2 * self.d_shift,
            width_size = d_hidden,
            depth      = 2,
            activation = jax.nn.tanh,
            key        = key,
        )

    def _params(
        self, x_pass: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        t_emb  = self.time_emb(t)
        raw = self.net(jnp.concatenate([x_pass, z_state, c, t_emb]))
        shift, raw_scale = jnp.split(raw, 2)
        return shift, self.alpha * jnp.tanh(raw_scale)

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward affine coupling transform."""
        if self.flip:
            eps_pass  = eps[self.d_half:]
            eps_shift = eps[:self.d_half]
            shift, log_scale = self._params(eps_pass, z_state, c, t)
            return jnp.concatenate([eps_shift * jnp.exp(log_scale) + shift, eps_pass])
        else:
            eps_pass  = eps[:self.d_half]
            eps_shift = eps[self.d_half:]
            shift, log_scale = self._params(eps_pass, z_state, c, t)
            return jnp.concatenate([eps_pass, eps_shift * jnp.exp(log_scale) + shift])

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Inverse affine coupling transform."""
        if self.flip:
            v_pass  = v[self.d_half:]
            v_shift = v[:self.d_half]
            shift, log_scale = self._params(v_pass, z_state, c, t)
            return jnp.concatenate([(v_shift - shift) * jnp.exp(-log_scale), v_pass])
        else:
            v_pass  = v[:self.d_half]
            v_shift = v[self.d_half:]
            shift, log_scale = self._params(v_pass, z_state, c, t)
            return jnp.concatenate([v_pass, (v_shift - shift) * jnp.exp(-log_scale)])

    def forward_log_det(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        eps_pass = eps[self.d_half:] if self.flip else eps[:self.d_half]
        _, log_scale = self._params(eps_pass, z_state, c, t)
        return jnp.sum(log_scale)


class RealNVPDiffusionNet(eqx.Module):
    """
    Conditional RealNVP transform for standard-Gaussian transition noise.
    """
    layers:        tuple
    d_x:   int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        n_layers:     int,
        time_emb_dim: int,
        key:          jr.PRNGKey,
        alpha:         float = 1.0,
    ):
        if alpha <= 0:
            raise ValueError(f"RealNVP alpha must be positive; got {alpha}.")
        self.d_x = d_x
        self.alpha = alpha
        keys = jr.split(key, n_layers)
        self.layers = tuple(
            RealNVPDiffusionLayer(
                d_x, d_c, d_hidden, time_emb_dim, flip=(i % 2 == 1),
                alpha=alpha, key=keys[i],
            )
            for i in range(n_layers)
        )

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """T_φ(eps; z_t, c_t, t)."""
        for layer in self.layers:
            eps = layer.apply(eps, z_state, c, t)
        return eps

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """T_φ^{-1}(v; z_t, c_t, t)."""
        for layer in reversed(self.layers):
            v = layer.apply_inverse(v, z_state, c, t)
        return v

    def inverse_and_log_det(self, v, z_state, c, t):
        log_det = jnp.array(0.0, dtype=v.dtype)
        current = v
        for layer in reversed(self.layers):
            eps = layer.apply_inverse(current, z_state, c, t)
            log_det += layer.forward_log_det(eps, z_state, c, t)  # ← minus
            current = eps
        return current, log_det

    def log_det(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        log_det = jnp.array(0.0, dtype=eps.dtype)
        current = eps
        for layer in self.layers:
            log_det = log_det + layer.forward_log_det(current, z_state, c, t)
            current = layer.apply(current, z_state, c, t)
        return log_det


class NICEDiffusionLayer(eqx.Module):
    """Conditional additive coupling with unit Jacobian determinant."""
    net: eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x: int = eqx.field(static=True)
    d_half: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)

    def __init__(self, d_x, d_c, d_hidden, time_emb_dim, flip, key):
        self.d_x = d_x
        self.d_half = d_x // 2
        self.flip = flip
        d_pass = d_x - self.d_half if flip else self.d_half
        d_shift = self.d_half if flip else d_x - self.d_half
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size=d_pass + d_x + d_c + time_emb_dim,
            out_size=d_shift,
            width_size=d_hidden,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def _shift(self, x_pass, z_state, c, t):
        return self.net(jnp.concatenate([x_pass, z_state, c, self.time_emb(t)]))

    def apply(self, x, z_state, c, t):
        if self.flip:
            x_pass, x_shift = x[self.d_half:], x[:self.d_half]
            return jnp.concatenate([x_shift + self._shift(x_pass, z_state, c, t), x_pass])
        x_pass, x_shift = x[:self.d_half], x[self.d_half:]
        return jnp.concatenate([x_pass, x_shift + self._shift(x_pass, z_state, c, t)])

    def apply_inverse(self, y, z_state, c, t):
        if self.flip:
            y_pass, y_shift = y[self.d_half:], y[:self.d_half]
            return jnp.concatenate([y_shift - self._shift(y_pass, z_state, c, t), y_pass])
        y_pass, y_shift = y[:self.d_half], y[self.d_half:]
        return jnp.concatenate([y_pass, y_shift - self._shift(y_pass, z_state, c, t)])


class NICEDiffusionNet(eqx.Module):
    """Volume-preserving NICE transform with a learnable global scale."""
    layers: tuple
    log_noise_scale: jnp.ndarray
    d_x: int = eqx.field(static=True)

    def __init__(self, d_x, d_c, d_hidden, n_layers, time_emb_dim, key,
                 noise_scale=0.5):
        if noise_scale <= 0:
            raise ValueError(f"NICE noise_scale must be positive; got {noise_scale}.")
        self.d_x = d_x
        self.log_noise_scale = jnp.asarray(jnp.log(noise_scale), dtype=jnp.float32)
        keys = jr.split(key, n_layers)
        self.layers = tuple(
            NICEDiffusionLayer(
                d_x, d_c, d_hidden, time_emb_dim, i % 2 == 1, keys[i],
            )
            for i in range(n_layers)
        )

    def noise_scale(self):
        return jnp.exp(self.log_noise_scale)

    def apply(self, eps, z_state, c, t):
        value = eps
        for layer in self.layers:
            value = layer.apply(value, z_state, c, t)
        return self.noise_scale() * value

    def apply_inverse(self, v, z_state, c, t):
        value = v / self.noise_scale()
        for layer in reversed(self.layers):
            value = layer.apply_inverse(value, z_state, c, t)
        return value

    def inverse_and_log_det(self, v, z_state, c, t):
        eps = self.apply_inverse(v, z_state, c, t)
        log_det = self.d_x * self.log_noise_scale.astype(v.dtype)
        return eps, log_det

    def log_det(self, eps, z_state, c, t):
        del z_state, c, t
        return self.d_x * self.log_noise_scale.astype(eps.dtype)

# ---------------------------------------------------------------------------
# Solver registry  (shared by RealNVPFlowSDE and helpers)
# ---------------------------------------------------------------------------

# Custom solver registry (our implementation)
_CUSTOM_SOLVER_CLASSES = {
    "euler":           EulerMaruyama,
    "heun":            StochasticHeun,
    "reversible_heun": ReversibleHeun,
    "milstein":        Milstein,
    "midpoint":        StochasticMidpoint,
    "adaptive_euler":  AdaptiveEulerMaruyama,
    "rk4":             StochasticRK4,
}


def _custom_solver_order(name: str) -> tuple[float, float]:
    """Return (weak/CDE order, strong order) for custom solver metadata."""
    name = name.lower()
    if name in ("heun", "reversible_heun", "milstein", "midpoint"):
        return 1.0, 1.0
    if name == "rk4":
        return 1.5, 1.5
    return 1.0, 0.5


def _make_custom_diffrax_solver(name: str):
    """Wrap a local stochastic solver so diffrax.diffeqsolve can drive it."""
    import diffrax

    if name not in _CUSTOM_SOLVER_CLASSES:
        raise ValueError(
            f"Unknown custom solver '{name}'. Choose from: {list(_CUSTOM_SOLVER_CLASSES)}"
        )

    class _DiffraxControlBridge:
        def __init__(self, term):
            self.term = term

        def evaluate(self, t0, t1):
            return self.term.contr(t0, t1)

    class _CustomDiffraxSolver(diffrax.AbstractItoSolver):
        base_solver: eqx.Module
        solver_name: str = eqx.field(static=True)

        term_structure = diffrax.AbstractTerm
        interpolation_cls = diffrax.LocalLinearInterpolation

        def order(self, terms):
            del terms
            order, _ = _custom_solver_order(self.solver_name)
            return order

        def strong_order(self, terms):
            del terms
            _, strong_order = _custom_solver_order(self.solver_name)
            return strong_order

        def error_order(self, terms):
            del terms
            return self.order(None)

        def init(self, terms, t0, t1, y0, args):
            del terms, t0, t1, y0, args
            return None

        def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del made_jump
            if not hasattr(terms, "terms") or len(terms.terms) != 2:
                raise ValueError(
                    "Custom diffrax solvers expect MultiTerm(ODETerm, ControlTerm)."
                )
            drift_term, diffusion_term = terms.terms
            control = _DiffraxControlBridge(diffusion_term)

            def drift_fn(t, z):
                return drift_term.vf(t, z, args)

            def diffuse_fn(t, z, dw):
                dt_sqrt = jnp.sqrt(_positive_dt(t0, t1))
                noise = dw / dt_sqrt
                return diffusion_term.prod(diffusion_term.vf(t, z, args), noise)

            y1 = self.base_solver.step(
                y0, t0, t1, drift_fn, diffuse_fn, control
            )
            dense_info = dict(y0=y0, y1=y1)
            return y1, None, dense_info, solver_state, diffrax.RESULTS.successful

    return _CustomDiffraxSolver(
        base_solver=_CUSTOM_SOLVER_CLASSES[name](),
        solver_name=name,
    )


def _get_diffrax_solver(name: str):
    """Return a diffrax-compatible SDE solver by name."""
    import diffrax
    name = name.lower()
    if name in _CUSTOM_SOLVER_CLASSES:
        return _make_custom_diffrax_solver(name)

    registry = {
        "euler":           diffrax.Euler(),
        "heun":            diffrax.Heun(),
        "reversible_heun": diffrax.ReversibleHeun(),
        "midpoint":        diffrax.Midpoint(),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown solver '{name}'. Choose from: {list(registry)}"
        )
    return registry[name]


# ---------------------------------------------------------------------------
# RealNVPFlowSDE — conditional RealNVP transition model
# ---------------------------------------------------------------------------

class RealNVPFlowSDE(eqx.Module):
    """
    Neural stochastic transition model with RealNVP-transformed Gaussian noise.

    Physical-time SDE:
        dz_t = f_θ(z_t, c_t, t) dt + G_φ(eps_t, z_t, c_t, t) \sqrt(dt)

    eps_t is drawn from N(0, I). G_φ is a bounded conditional RealNVP.

    Discretised update at each solver step:
        z_{t+dt} = z_t + f_drift(z_t) dt + G_φ(eps_t, z_t, c_t, t) \sqrt(dt)

    where G_φ(eps_t, z_t, c_t, t) \sqrt(dt) is the shaped diffusion contribution.

    Training
    --------
    transition NLL — invert observed one-step residuals through RealNVP and
                     minimise Gaussian latent NLL plus the exact log determinant.

    Supported solvers
    -----------------
    "euler"           — Euler-Maruyama       (strong order 0.5)
    "heun"            — Heun                 (strong order 1.0, additive noise)
    "reversible_heun" — ReversibleHeun       (memory-efficient reversible)
    "midpoint"        — Stochastic midpoint
    """
    encoder:    BiGRUEncoder
    init_net:   GRUInitialInference
    drift:      SDEDriftNet
    diffusion:  RealNVPDiffusionNet | NICEDiffusionNet
    d_x:        int    = eqx.field(static=True)
    d_y:        int    = eqx.field(static=True)
    d_u:        int    = eqx.field(static=True)
    exclude_control_from_diffusion: bool = eqx.field(static=True)
    physics_fn: object = eqx.field(static=True, default=None)
    flow_type:  str = eqx.field(static=True, default="realnvp")
    observation_stride: int = eqx.field(static=True, default=1)
    time_step: float = eqx.field(static=True, default=1.0)

    def __init__(
        self,
        d_x:           int,
        d_y:           int,
        d_u:           int   = 0,
        exclude_control_from_diffusion: bool = False,
        d_c:           int   = 64,
        d_hidden_enc:  int   = 64,
        d_hidden:      int   = 128,
        d_hidden_diffusion: int = 64,
        depth:         int   = 3,
        time_emb_dim:  int   = 32,
        n_realnvp_layers: int = 2,
        realnvp_alpha: float = 1.0,
        flow_type:      str = "realnvp",
        n_nice_layers:  int = 2,
        nice_noise_scale: float = 0.5,
        observation_stride: int = 1,
        time_step:      float = 1.0,
        physics_fn            = None,
        key:           jr.PRNGKey = jr.PRNGKey(0),
    ):
        self.d_x        = d_x
        self.d_y        = d_y
        self.d_u        = d_u
        self.exclude_control_from_diffusion = exclude_control_from_diffusion
        if d_u < 0 or d_u > d_y:
            raise ValueError(f"d_u must be between 0 and d_y; got d_u={d_u}, d_y={d_y}.")
        self.physics_fn = physics_fn
        self.flow_type = flow_type.lower()
        if observation_stride < 1:
            raise ValueError("observation_stride must be at least 1.")
        if time_step <= 0:
            raise ValueError("time_step must be positive.")
        self.observation_stride = observation_stride
        self.time_step = float(time_step)
        k_enc, k_init, k_drift, k_diff = jr.split(key, 4)
        self.encoder   = BiGRUEncoder(d_y, d_hidden_enc, d_c, k_enc)
        self.init_net  = GRUInitialInference(d_y, d_hidden_enc, d_x, k_init)
        self.drift     = SDEDriftNet(d_x, d_c, d_hidden, depth, time_emb_dim, k_drift)
        if self.flow_type == "realnvp":
            self.diffusion = RealNVPDiffusionNet(
                d_x, d_c, d_hidden_diffusion, n_realnvp_layers, time_emb_dim, k_diff,
                alpha=realnvp_alpha,
            )
        elif self.flow_type == "nice":
            self.diffusion = NICEDiffusionNet(
                d_x, d_c, d_hidden_diffusion, n_nice_layers, time_emb_dim, k_diff,
                noise_scale=nice_noise_scale,
            )
        else:
            raise ValueError(f"Unknown flow_type '{flow_type}'; choose 'realnvp' or 'nice'.")

    def encode(self, y: jnp.ndarray) -> jnp.ndarray:
        """y: (T, d_y) → c: (T, d_c)."""
        return self.encoder(y)

    def encode_observation_stride(self, y: jnp.ndarray, stride: int) -> jnp.ndarray:
        """Encode observations every ``stride`` steps and hold context between them."""
        if stride < 1:
            raise ValueError(f"observation stride must be >= 1; got {stride}.")
        sparse_context = self.encode(y[::stride])
        context_idx = jnp.minimum(
            jnp.arange(y.shape[0]) // stride, sparse_context.shape[0] - 1
        )
        return sparse_context[context_idx]

    def encode_diffusion_stride(self, y: jnp.ndarray, stride: int) -> jnp.ndarray:
        """Encode stochastic context with control channels masked when requested."""
        if self.exclude_control_from_diffusion and self.d_u:
            y = y.at[..., -self.d_u:].set(0.0)
        return self.encode_observation_stride(y, stride)

    def infer_z0(
        self,
        y_seg: jnp.ndarray,
        key:   jr.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample z0 with the reparameterisation trick and return its parameters."""
        mu, sigma = self.init_net(y_seg)
        eps = jr.normal(key, mu.shape)
        z0 = mu + sigma * eps
        return z0, mu, sigma, jnp.array(0.0, dtype=mu.dtype)

    @staticmethod
    def initial_state_nll(
        x0: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        """Mean negative log likelihood of x0 under a diagonal Gaussian."""
        standardized = (x0 - mu) / sigma
        return jnp.mean(
            0.5 * standardized**2 + jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def transition_nll(self, x_seg, c_drift_seg, c_diffusion_seg, t_start, dt):
        times = t_start + jnp.arange(x_seg.shape[0] - 1) * dt

        def one_step(x0, x1, c_drift0, c_diffusion0, t0):
            drift = self.drift(x0, c_drift0, t0)
            if self.physics_fn is not None:
                drift = drift + self.physics_fn(x0)
            normalized_residual = (x1 - x0 - drift * dt) / jnp.sqrt(dt)
            eps, log_det = self.diffusion.inverse_and_log_det(
                normalized_residual, x0, c_diffusion0, t0
            )
            base_nll = 0.5 * jnp.sum(eps**2) + 0.5 * self.d_x * jnp.log(2.0 * jnp.pi)
            time_log_det = 0.5 * self.d_x * jnp.log(dt)   # ← FIXED sign
            # log_det from inverse_and_log_det is now -log|det dT/deps| → correct
            return (base_nll + log_det + time_log_det) / self.d_x

        losses = jax.vmap(one_step)(
            x_seg[:-1], x_seg[1:], c_drift_seg[:-1], c_diffusion_seg[:-1], times
        )
        return jnp.mean(losses)

    # ------------------------------------------------------------------
    # Pathwise training:  integrate SDE → minimise reconstruction loss
    # ------------------------------------------------------------------

    def _build_terms(self, c_interp, diffusion_scale: float = 1.0):
        del c_interp, diffusion_scale
        raise NotImplementedError(
            "A nonlinear RealNVP noise transform cannot be represented as a "
            "Diffrax diffusion matrix. Use the explicit custom solver."
        )

    def _integrate_segment(
        self,
        x_seg:       jnp.ndarray,   # (L+1, d_x)  ground-truth segment
        z0:          jnp.ndarray,   # (d_x,)  integration initial state
        c_interp,
        t0_seg:      float,
        t1_seg:      float,
        ts_seg:      jnp.ndarray,   # (L+1,)  save times
        key:         jr.PRNGKey,
        dt0:         float,
        solver_name: str,
    ) -> jnp.ndarray:
        """
        Integrate the RealNVP SDE over one segment and return the predicted path.

        eps_t ~ N(0, I) is the latent noise for each custom solver step.
        """
        import diffrax
        drift_fn, diff_fn = self._build_terms(c_interp)
        solver = _get_diffrax_solver(solver_name)
        bm = diffrax.VirtualBrownianTree(
            t0=t0_seg, t1=t1_seg + dt0, tol=dt0 * 0.5,
            shape=(self.d_x,), key=key,
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn),
            diffrax.ControlTerm(diff_fn, bm),
        )
        sol = diffrax.diffeqsolve(
            terms, solver,
            t0=t0_seg, t1=t1_seg, dt0=dt0,
            y0=z0,
            saveat=diffrax.SaveAt(ts=ts_seg),
        )
        return sol.ys

    def nll_pathwise_single(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        seg_length:       int,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
        shooting_noise_std: float = 0.0,
        kl_weight:        float = 1e-3,
        continuity_weight: float = 0.0,
        init_loss_weight: float = 1.0,
        infer_initial:    bool = True,
    ) -> jnp.ndarray:
        """
        Path-wise multiple-shooting reconstruction loss for one trajectory.

        Each segment starts from either a sampled GRU-inferred state or its
        provided ground-truth state, as selected by ``infer_initial``.
        A fresh noise key is drawn per segment; G_φ shapes standard Gaussian eps.
        Gradients flow back through the solver via the adjoint method.
        
        Args:
            use_custom_solver: If True, use custom stochastic solvers instead of diffrax.
            shooting_noise_std: Stddev of normalized-state noise added to non-initial
                segment starts. This trains recovery from rollout-induced errors.
        """
        if use_custom_solver:
            return self._nll_pathwise_single_custom(
                x, y, key, seg_length, solver_name, shooting_noise_std,
                kl_weight, continuity_weight, init_loss_weight, infer_initial
            )
        else:
            raise ValueError(
                "RealNVP transition noise is nonlinear in eps and requires "
                "--use_custom_solver; it cannot be represented by a Diffrax ControlTerm."
            )

    def _nll_pathwise_single_diffrax(
        self,
        x:           jnp.ndarray,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        seg_length:  int,
        solver_name: str = "heun",
        shooting_noise_std: float = 0.0,
        kl_weight: float = 1e-3,
        continuity_weight: float = 0.0,
        init_loss_weight: float = 1.0,
        infer_initial: bool = True,
    ) -> jnp.ndarray:
        """Path-wise loss using diffrax solvers."""
        import diffrax
        T        = x.shape[0]
        dt0      = self.time_step
        c        = self.encode_observation_stride(y, self.observation_stride)
        ts_all   = jnp.arange(1, T + 1) * dt0
        c_interp = diffrax.LinearInterpolation(ts=ts_all, ys=c)

        n_segs   = (T - 1) // seg_length
        seg_keys = jr.split(key, n_segs)
        init_keys = jr.split(jr.fold_in(key, 17), n_segs)
        total_recon = jnp.array(0.0)
        total_kl    = jnp.array(0.0)
        total_cont  = jnp.array(0.0)
        total_init  = jnp.array(0.0)
        prev_end    = None
        for k in range(n_segs):
            s      = k * seg_length
            e      = s + seg_length + 1
            t0_seg = (s + 1) * dt0
            t1_seg = e * dt0
            ts_seg = jnp.linspace(t0_seg, t1_seg, seg_length + 1)
            if infer_initial:
                z0, mu0, sigma0, kl = self.infer_z0(y[s:e], init_keys[k])
                total_init = total_init + self.initial_state_nll(x[s], mu0, sigma0)
            else:
                z0, mu0, kl = x[s], x[s], jnp.array(0.0)
            z0 = z0 + shooting_noise_std * jr.normal(jr.fold_in(init_keys[k], 101), z0.shape)

            pred = self._integrate_segment(
                x[s:e], z0, c_interp, t0_seg, t1_seg, ts_seg,
                seg_keys[k], dt0, solver_name,
            )
            total_recon = total_recon + jnp.mean((pred - x[s:e]) ** 2)
            total_kl = total_kl + kl
            if prev_end is not None:
                total_cont = total_cont + jnp.mean((prev_end - mu0) ** 2)
            prev_end = pred[-1]

        denom_cont = jnp.maximum(n_segs - 1, 1)
        recon = total_recon / n_segs
        kl = total_kl / n_segs
        cont = total_cont / denom_cont
        init = total_init / n_segs
        return recon + kl_weight * kl + continuity_weight * cont + init_loss_weight * init

    def _nll_pathwise_single_custom(
            self, x, y, key, seg_length, solver_name="heun",
            shooting_noise_std=0.0, kl_weight=1e-3,
            continuity_weight=0.0, init_loss_weight=1.0,
            infer_initial=True):
        T   = x.shape[0]
        dt0 = self.time_step
        c   = self.encode_observation_stride(y, self.observation_stride)
        c_diffusion = self.encode_diffusion_stride(y, self.observation_stride)

        n_segs   = (T - 1) // seg_length
        k_seg_root, k_init_root = jr.split(key, 2)        # ← independent keys
        seg_keys  = jr.split(k_seg_root,  n_segs)
        init_keys = jr.split(k_init_root, n_segs)

        total_recon = jnp.array(0.0)
        total_init  = jnp.array(0.0)
        total_cont  = jnp.array(0.0)
        prev_end    = None

        for k in range(n_segs):
            s      = k * seg_length
            e      = s + seg_length + 1
            t0_seg = s * dt0
            t1_seg = (e - 1) * dt0
            ts_seg = jnp.linspace(t0_seg, t1_seg, seg_length + 1)

            if infer_initial:
                z0, mu0, sigma0, _ = self.infer_z0(y[s:e], init_keys[k])
                total_init = total_init + self.initial_state_nll(x[s], mu0, sigma0)
            else:
                z0, mu0 = x[s], x[s]

            if shooting_noise_std > 0.0 and k > 0:
                z0 = z0 + shooting_noise_std * jr.normal(jr.fold_in(init_keys[k], 101), z0.shape)

            # ← Use x_pred for the reconstruction loss (was discarded before)
            x_pred = self._integrate_custom(
                z0, ts_seg, c, seg_keys[k], solver_name,
                c_diffusion=c_diffusion,
            )
            total_recon = total_recon + jnp.mean((x_pred - x[s:e]) ** 2)

            if prev_end is not None and continuity_weight > 0.0:
                total_cont = total_cont + jnp.mean((prev_end - mu0) ** 2)
            prev_end = x_pred[-1]

        n = n_segs
        denom_cont = jnp.maximum(n - 1, 1)
        return (total_recon / n
                + init_loss_weight * total_init / n
                + continuity_weight * total_cont / denom_cont)

    def _context_at(self, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Nearest-neighbour context lookup on the physical-time grid."""
        T = c.shape[0]
        # c[i] describes x[i + 1], located at physical time (i + 1) * time_step.
        idx = jnp.rint(t / self.time_step - 1.0).astype(jnp.int32)
        idx = jnp.clip(idx, 0, T - 1)
        return c[idx]

    def _custom_drift(self, z: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        c_t = self._context_at(c, t)
        f = self.drift(z, c_t, t)
        if self.physics_fn is not None:
            f = f + self.physics_fn(z)
        return f

    def _custom_diffusion(
        self,
        eps:            jnp.ndarray,
        z:              jnp.ndarray,
        c:              jnp.ndarray,
        t:              jnp.ndarray,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        c_t = self._context_at(c, t)
        return self.diffusion.apply(eps * diffusion_temp, z, c_t, t)

    def _integrate_custom(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
        c_diffusion:    jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Backward-compatible alias for the shared stochastic solver path."""
        return self._integrate_stochastic_solver(
            z0, ts, c, key, solver_name, diffusion_temp, c_diffusion
        )

    def _integrate_stochastic_solver(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
        c_diffusion:    jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Integrate using solver implementations from stochastic_solvers.py."""
        solver = get_stochastic_solver(solver_name)
        if c_diffusion is None:
            c_diffusion = c
        eps_steps = jr.normal(key, (ts.shape[0] - 1, self.d_x))
        dt0 = ts[1] - ts[0]
        bm = StepNoisePath(eps_steps, ts[0], dt0)

        def drift_fn(t, z):
            return self._custom_drift(z, c, t)

        def diffuse_fn(t, z, eps):
            return self._custom_diffusion(eps, z, c_diffusion, t, diffusion_temp)

        def step(z, t_pair):
            t0, t1 = t_pair
            z1 = solver.step(z, t0, t1, drift_fn, diffuse_fn, bm)
            return z1, z1

        _, states = jax.lax.scan(step, z0, (ts[:-1], ts[1:]))
        return jnp.concatenate([z0[None], states], axis=0)

    def nll_pathwise(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        seg_length:       int,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
        shooting_noise_std: float = 0.0,
        kl_weight:        float = 1e-3,
        continuity_weight: float = 0.0,
        init_loss_weight: float = 1.0,
        infer_initial:    bool = True,
    ) -> jnp.ndarray:
        """Batch path-wise reconstruction loss. x: (B,T,d_x)  y: (B,T,d_y)."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.nll_pathwise_single(
                xi, yi, ki, seg_length, solver_name, use_custom_solver,
                shooting_noise_std, kl_weight, continuity_weight, init_loss_weight,
                infer_initial
            )
        )(x, y, keys))

    def rollout_loss_single(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
        infer_initial: bool = True,
        num_ensemble: int = 20,
        loss_type: str = "gaussian_nll",
        energy_beta: float = 1.0,
        energy_uncertainty_weight: float = 0.5,
    ) -> jnp.ndarray:
        """Full-horizon rollout loss from x[0].

        ``loss_type="mse"`` keeps the old one-sample reconstruction objective.
        ``"energy_score"`` and ``"gaussian_nll"`` evaluate an ensemble of SDE
        paths, giving the diffusion model a proper probabilistic training signal.
        """
        T = x.shape[0]
        dt0 = self.time_step
        ts = jnp.arange(T, dtype=x.dtype) * dt0
        c = self.encode_observation_stride(y, self.observation_stride)
        c_diffusion = self.encode_diffusion_stride(y, self.observation_stride)

        init_key, sde_key = jr.split(key)
        if loss_type == "mse":
            if infer_initial:
                z0, _, _, _ = self.infer_z0(y, init_key)
            else:
                z0 = x[0]
            if not use_custom_solver:
                raise ValueError(
                    "RealNVP rollout loss requires --use_custom_solver; "
                    "Diffrax ControlTerm cannot represent nonlinear flow noise."
                )
            pred = self._integrate_stochastic_solver(
                z0, ts, c, sde_key, solver_name, c_diffusion=c_diffusion
            )
            return jnp.mean((pred - x) ** 2)

        if not use_custom_solver:
            raise ValueError(
                "Probabilistic RealNVP rollout loss requires --use_custom_solver; "
                "Diffrax ControlTerm cannot represent nonlinear flow noise."
            )

        if infer_initial:
            init_keys = jr.split(init_key, num_ensemble)
            z0s = jax.vmap(lambda k: self.infer_z0(y, k)[0])(init_keys)
        else:
            z0s = jnp.broadcast_to(x[0], (num_ensemble, self.d_x))

        sde_keys = jr.split(sde_key, num_ensemble)

        def integrate_one(z0, k):
            return self._integrate_stochastic_solver(
                z0, ts, c, k, solver_name, c_diffusion=c_diffusion
            )

        preds = jax.vmap(integrate_one)(z0s, sde_keys)
        if loss_type == "energy_score":
            return energy_score(
                preds,
                x,
                beta=energy_beta,
                uncertainty_weight=energy_uncertainty_weight,
            )
        if loss_type == "gaussian_nll":
            return gaussian_nll(preds, x)
        raise ValueError(
            f"Unknown rollout loss_type '{loss_type}'. "
            "Choose 'mse', 'energy_score', or 'gaussian_nll'."
        )

    def rollout_loss(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
        infer_initial: bool = True,
        num_ensemble: int = 20,
        loss_type: str = "energy_score",
        energy_beta: float = 1.0,
        energy_uncertainty_weight: float = 0.5,
        rollout_window: int = 0,
    ) -> jnp.ndarray:
        """Batch full-horizon pathwise rollout loss."""
        if rollout_window > 0 and rollout_window < x.shape[1]:
            max_start = x.shape[1] - rollout_window
            start = jr.randint(jr.fold_in(key, 503), (), 0, max_start + 1)
            x = jax.lax.dynamic_slice_in_dim(x, start, rollout_window, axis=1)
            y = jax.lax.dynamic_slice_in_dim(y, start, rollout_window, axis=1)
        B = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.rollout_loss_single(
                xi, yi, ki, solver_name, use_custom_solver, infer_initial,
                num_ensemble, loss_type, energy_beta, energy_uncertainty_weight
            )
        )(x, y, keys))

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x0:             jnp.ndarray,   # (d_x,)
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        solver_name:    str   = "heun",
        dt_factor:      float | None = None,
        diffusion_temp: float = 1.0,
        infer_initial:  bool  = False,
    ) -> jnp.ndarray:
        """
        Sample p(x_{1:T} | y_{1:T}) via the RealNVP SDE.

        By default every particle starts at the supplied ``x0``. Set
        ``infer_initial=True`` to instead sample each particle's initial state
        from the learned diagonal Gaussian conditioned on ``y``.
        G_φ(z_t, c_t, t) shapes standard Gaussian eps into structured diffusion.
        Returns (n_particles, T, d_x).
        """
        import diffrax
        T   = y.shape[0]
        duration = T * self.time_step if dt_factor is None else dt_factor
        dt0 = duration / T
        c = self.encode_observation_stride(y, self.observation_stride)
        ts  = jnp.linspace(0.0, duration, T + 1)
        c_ts = jnp.linspace(dt0, duration, T)
        c_interp          = diffrax.LinearInterpolation(ts=c_ts, ys=c)
        drift_fn, diff_fn = self._build_terms(c_interp, diffusion_temp)
        solver            = _get_diffrax_solver(solver_name)
        saveat            = diffrax.SaveAt(ts=ts)

        def simulate_one(k):
            k_init, k_sde = jr.split(k)
            if infer_initial:
                z0, _, _, _ = self.infer_z0(y, k_init)
            else:
                z0 = x0
            bm = diffrax.VirtualBrownianTree(
                t0=0.0, t1=duration + dt0, tol=dt0 * 0.5,
                shape=(self.d_x,), key=k_sde,
            )
            terms = diffrax.MultiTerm(
                diffrax.ODETerm(drift_fn),
                diffrax.ControlTerm(diff_fn, bm),
            )
            sol = diffrax.diffeqsolve(
                terms, solver,
                t0=0.0, t1=duration, dt0=dt0, y0=z0, saveat=saveat,
            )
            return sol.ys[1:]   # (T, d_x), excludes supplied x0

        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    def sample_posterior_custom_solver(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x0:             jnp.ndarray,   # (d_x,)
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        solver_name:    str   = "heun",
        dt_factor:      float | None = None,
        diffusion_temp: float = 1.0,
        infer_initial:  bool  = False,
        observation_stride: int | None = None,
    ) -> jnp.ndarray:
        """
        Sample p(x_{1:T} | y_{1:T}) using custom stochastic solvers.
        
        This method uses our custom SDE solvers (Euler-Maruyama, Heun, Milstein, etc.)
        instead of diffrax, allowing for more control over the integration and
        better integration with the RealNVP flow diffusion model.
        
        The custom solver form is:
            dz_t = f_θ(z_t, c_t, t) dt + G_φ(eps_t, z_t, c_t, t) sqrt(dt)

        where G_φ is the RealNVP flow that transforms eps_t ~ N(0, I).
        
        Args:
            y: Observations (T, d_y)
            x0: Initial state (d_x,)
            key: Random key
            n_particles: Number of particles to sample
            solver_name: Custom solver name ("euler", "heun", "milstein", "midpoint", etc.)
            dt_factor: Optional total duration override. By default, physical
                time is determined by ``T * self.time_step``.
            diffusion_temp: Temperature scaling for diffusion
            infer_initial: Sample the initial state from the learned inference
                network instead of using the supplied x0. Defaults to False.
        
        Returns:
            Samples (n_particles, T, d_x)
        """
        T   = y.shape[0]
        stride = self.observation_stride if observation_stride is None else observation_stride
        c = self.encode_observation_stride(y, stride)
        c_diffusion = self.encode_diffusion_stride(y, stride)
        # y and the returned trajectory correspond to x[1:T+1], while x0 is
        # the state immediately before them. Integrate one additional interval
        # and discard the supplied initial state from the returned path.
        duration = T * self.time_step if dt_factor is None else dt_factor
        ts  = jnp.linspace(0.0, duration, T + 1)
        
        def simulate_one(particle_key):
            k_init, k_sde = jr.split(particle_key)
            if infer_initial:
                z0, _, _, _ = self.infer_z0(y, k_init)
            else:
                z0 = x0
            path = self._integrate_custom(
                z0, ts, c, k_sde, solver_name, diffusion_temp,
                c_diffusion=c_diffusion,
            )
            return path[1:]
        
        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    def sample_posterior_batch(
        self,
        y:           jnp.ndarray,   # (B, T, d_y)
        x0:          jnp.ndarray,   # (B, d_x)
        key:         jr.PRNGKey,
        n_particles: int = 50,
        solver_name: str = "heun",
    ) -> jnp.ndarray:
        """Returns (n_particles, B, T, d_x)."""
        B    = y.shape[0]
        keys = jr.split(key, B)
        samples = jax.vmap(
            lambda yi, x0i, ki: self.sample_posterior(
                yi, x0i, ki, n_particles, solver_name
            )
        )(y, x0, keys)
        return jnp.swapaxes(samples, 0, 1)

# ---------------------------------------------------------------------------
# RealNVPFlowSDE training utilities
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step_realnvp_pathwise(
    model:            RealNVPFlowSDE,
    x_batch:          jnp.ndarray,
    y_batch:          jnp.ndarray,
    key:              jr.PRNGKey,
    seg_length:       int,
    opt_state,
    optimizer,
    solver_name:      str = "heun",
    use_custom_solver: bool = False,
    shooting_noise_std: float = 0.0,
    rollout_loss_weight: float = 0.0,
    kl_weight: float = 1e-3,
    continuity_weight: float = 0.0,
    init_loss_weight: float = 1.0,
    infer_initial: bool = True,
    rollout_num_ensemble: int = 20,
    rollout_loss_type: str = "energy_score",
    rollout_energy_beta: float = 1.0,
    rollout_uncertainty_weight: float = 0.5,
    rollout_window: int = 0,
    transition_loss_weight: float = 0.0,
):
    """
    Path-wise training step.

    Integrates the RealNVP SDE for each multiple-shooting segment.
    eps_t ~ N(0, I) per solver step. G_φ shapes eps_t; gradients flow through
    the solver.
    
    Args:
        use_custom_solver: If True, use custom stochastic solvers from stochastic_solvers.py
                          instead of diffrax. Supports: euler, heun, reversible_heun, 
                          midpoint, milstein, adaptive_euler, rk4.
    """
    def loss_fn(m):
        path_loss = m.nll_pathwise(
            x_batch, y_batch, key, seg_length, solver_name, use_custom_solver,
            shooting_noise_std, kl_weight, continuity_weight, init_loss_weight,
            infer_initial
        )
        total_loss = path_loss
        if transition_loss_weight != 0.0:
            def one_transition_nll(xi, yi):
                drift_context = m.encode_observation_stride(
                    yi, m.observation_stride
                )
                diffusion_context = m.encode_diffusion_stride(
                    yi, m.observation_stride
                )
                return m.transition_nll(
                    xi, drift_context, diffusion_context,
                    jnp.asarray(0.0, xi.dtype), m.time_step
                )

            transition_loss = jnp.mean(
                jax.vmap(one_transition_nll)(x_batch, y_batch)
            )
            total_loss = total_loss + transition_loss_weight * transition_loss
        if rollout_loss_weight != 0.0:
            rollout_loss = m.rollout_loss(
                x_batch, y_batch, jr.fold_in(key, 29), solver_name,
                use_custom_solver, infer_initial, rollout_num_ensemble,
                rollout_loss_type, rollout_energy_beta,
                rollout_uncertainty_weight, rollout_window
            )
            total_loss = total_loss + rollout_loss_weight * rollout_loss
        return total_loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss
