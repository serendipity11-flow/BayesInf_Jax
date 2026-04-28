"""
ConditionalNeuralSDE / NICEFlowSDE: Flow-SDE for posterior trajectory estimation.

Model:
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(z_t, c_t, t) dW_t

    c_t = BiGRUEncoder(y_{1:T})_t  — per-step context from a bidirectional GRU
          over the entire observation sequence (smoothing: c_t sees future y's).

    f_θ — neural drift (MLP). To add a physics prior, wrap externally:
          f_total = physics_fn(z_t, t) + model.drift(z_t, c_t, t).

    G_φ — NICE-structured state-dependent diffusion factor:
              G_k = [[I,   0 ], [M_k, I]]  (flip=False)
              G_k = [[I,  M_k], [0,   I]]  (flip=True)
          G_φ = G_L ∘ … ∘ G_1  (composition of L alternating coupling layers)
          M_k : R^{d_pass + d_c + t_emb} → R^{d_shift × d_pass}  (matrix-valued MLP)

          det(G_φ) = 1 for every state — variance collapse is structurally
          impossible.  Q = G_φ G_φ^T is always positive-definite.

Training — Pathwise MLE under Euler-Maruyama discretization:
    L = Σ_t  0.5 ||G_φ^{-1}(Δx_t − f_t dt)||² / dt
    No log-det term because det(G_φ) = 1 always.

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

# Import custom stochastic solvers
from bayesinf.applications.state_estimation.stochastic_solvers import (
    EulerMaruyama,
    StochasticHeun,
    ReversibleHeun,
    Milstein,
    StochasticMidpoint,
    AdaptiveEulerMaruyama,
    StochasticRK4,
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


# ---------------------------------------------------------------------------
# Drift network  f_θ(z_t, c_t, t)
# ---------------------------------------------------------------------------

class SDEDriftNet(eqx.Module):
    """
    Parameterises the SDE drift  f_θ(z_t, c_t, t) → (d_x,).
    Input:  z_t  (d_x,)  |  c_t  (d_c,)  |  t  scalar ∈ [0, 1]
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
    ):
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size    = d_x + d_c + time_emb_dim,
            out_size   = d_x,
            width_size = d_hidden,
            depth      = depth,
            activation = jax.nn.gelu,
            key        = key,
        )

    def __call__(self, z: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t_emb = self.time_emb(t)
        return self.net(jnp.concatenate([z, c, t_emb]))


# ---------------------------------------------------------------------------
# NICE diffusion network  G_φ(z_t, c_t, t)
# ---------------------------------------------------------------------------

class NICEDiffusionLayer(eqx.Module):
    """
    One NICE coupling layer used as a factor of the diffusion matrix G.

    Structure (flip=False):
        G_k = [[I_{d1},  0  ],
               [M_k,  I_{d2}]]
        d1 = d_x // 2  (passthrough rows/cols)
        d2 = d_x − d1  (shifted rows/cols)

    Structure (flip=True):
        G_k = [[I_{d1}, M_k],
               [0,   I_{d2}]]
        d1 = d_x // 2  (shifted rows/cols)
        d2 = d_x − d1  (passthrough rows/cols)

    M_k : R^{d_pass + d_c + t_emb} → R^{d_shift × d_pass}  (matrix-valued MLP)
    det(G_k) = 1 by block-triangular structure with identity diagonal blocks.

    apply         : G_k @ eps  (forward — used in sampling)
    apply_inverse : G_k^{-1} @ v  (used in NLL computation)
    as_matrix     : returns explicit (d_x, d_x) matrix G_k
    """
    net:     eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x:     int  = eqx.field(static=True)
    d_half:  int  = eqx.field(static=True)
    d_pass:  int  = eqx.field(static=True)
    d_shift: int  = eqx.field(static=True)
    flip:    bool = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        time_emb_dim: int,
        flip:         bool,
        key:          jr.PRNGKey,
    ):
        self.d_x    = d_x
        self.d_half = d_x // 2
        self.flip   = flip
        if flip:
            self.d_pass  = d_x - d_x // 2   # lower half of state
            self.d_shift = d_x // 2          # upper half
        else:
            self.d_pass  = d_x // 2          # upper half of state
            self.d_shift = d_x - d_x // 2    # lower half
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size    = self.d_pass + d_c + time_emb_dim,
            out_size   = self.d_shift * self.d_pass,
            width_size = d_hidden,
            depth      = 2,
            activation = jax.nn.tanh,
            key        = key,
        )

    def _get_M(self, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the (d_shift, d_pass) coupling matrix from physical state.
        tanh bounds entries to (-1, 1) — prevents G from exploding when z_state
        drifts outside the training distribution during SDE integration.
        """
        z_pass = z_state[self.d_half:] if self.flip else z_state[:self.d_half]
        t_emb  = self.time_emb(t)
        raw    = self.net(jnp.concatenate([z_pass, c, t_emb]))
        return jax.nn.tanh(raw).reshape(self.d_shift, self.d_pass)

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """G_k @ eps  (M depends on z_state, not on eps)."""
        M = self._get_M(z_state, c, t)
        if self.flip:
            eps_pass  = eps[self.d_half:]
            eps_shift = eps[:self.d_half]
            return jnp.concatenate([M @ eps_pass + eps_shift, eps_pass])
        else:
            eps_pass  = eps[:self.d_half]
            eps_shift = eps[self.d_half:]
            return jnp.concatenate([eps_pass, M @ eps_pass + eps_shift])

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """G_k^{-1} @ v  (inverse of block-triangular layer)."""
        M = self._get_M(z_state, c, t)
        if self.flip:
            v_pass  = v[self.d_half:]
            v_shift = v[:self.d_half]
            return jnp.concatenate([v_shift - M @ v_pass, v_pass])
        else:
            v_pass  = v[:self.d_half]
            v_shift = v[self.d_half:]
            return jnp.concatenate([v_pass, v_shift - M @ v_pass])

    def as_matrix(self, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Explicit (d_x, d_x) matrix for G_k."""
        M   = self._get_M(z_state, c, t)
        G_k = jnp.eye(self.d_x)
        if self.flip:
            G_k = G_k.at[:self.d_half, self.d_half:].set(M)
        else:
            G_k = G_k.at[self.d_half:, :self.d_half].set(M)
        return G_k


class NICEDiffusionNet(eqx.Module):
    """
    NICE-structured state-dependent diffusion matrix G_φ(z_t, c_t, t).

    G_φ = G_L ∘ … ∘ G_1   (composition of alternating coupling layers)

    Each G_k has det = 1  →  det(G_φ) = 1 always.
    Variance collapse is structurally impossible: G cannot shrink to zero
    because the diagonal blocks are always the identity.

    The SDE covariance Q = G_φ G_φ^T is always positive-definite.
    """
    layers: tuple
    d_x:    int = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        n_layers:     int,
        time_emb_dim: int,
        key:          jr.PRNGKey,
    ):
        self.d_x = d_x
        keys = jr.split(key, n_layers)
        self.layers = tuple(
            NICEDiffusionLayer(d_x, d_c, d_hidden, time_emb_dim, flip=(i % 2 == 1), key=keys[i])
            for i in range(n_layers)
        )

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """G_φ(z_t) @ eps — forward application to a noise vector."""
        for layer in self.layers:
            eps = layer.apply(eps, z_state, c, t)
        return eps

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """G_φ(z_t)^{-1} @ v — used in NLL computation."""
        for layer in reversed(self.layers):
            v = layer.apply_inverse(v, z_state, c, t)
        return v

    def matrix(self, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Construct the explicit (d_x, d_x) matrix G_φ(z_t).
        Used by diffrax ControlTerm in the SDE solvers.
        """
        G = jnp.eye(self.d_x)
        for layer in self.layers:
            G = layer.as_matrix(z_state, c, t) @ G
        return G

# ---------------------------------------------------------------------------
# Solver registry  (shared by NICEFlowSDE and helpers)
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


def _get_solver(name: str, use_custom: bool = True):
    """
    Return a stochastic solver by name.
    
    Args:
        name: Solver name
        use_custom: If True, prefer custom solvers over diffrax
    
    Returns:
        Solver instance (custom solver or diffrax solver)
    
    Supported custom solvers (our implementation):
        - "euler"           — Euler-Maruyama       (strong order 0.5)
        - "heun"            — Stochastic Heun      (strong order 1.0)
        - "reversible_heun" — Reversible Heun      (memory-efficient)
        - "milstein"        — Milstein             (strong order 1.0)
        - "midpoint"        — Stochastic Midpoint  (improved stability)
        - "adaptive_euler"  — Adaptive Euler       (error-based step size)
        - "rk4"             — Stochastic RK4       (higher order)
    
    Supported diffrax solvers (fallback):
        - "euler"           — diffrax.Euler()
        - "heun"            — diffrax.Heun()
        - "reversible_heun" — diffrax.ReversibleHeun()
        - "midpoint"        — diffrax.Midpoint()
    """
    name_lower = name.lower()
    
    # Try custom solver first
    if use_custom and name_lower in _CUSTOM_SOLVER_CLASSES:
        return _CUSTOM_SOLVER_CLASSES[name_lower]()
    
    # Fall back to diffrax
    import diffrax
    diffrax_registry = {
        "euler":           diffrax.Euler(),
        "heun":            diffrax.Heun(),
        "reversible_heun": diffrax.ReversibleHeun(),
        "midpoint":        diffrax.Midpoint(),
    }
    if name_lower in diffrax_registry:
        return diffrax_registry[name_lower]
    
    raise ValueError(
        f"Unknown solver '{name}'. Available custom: {list(_CUSTOM_SOLVER_CLASSES.keys())}, "
        f"available diffrax: {list(diffrax_registry.keys())}"
    )


def _get_diffrax_solver(name: str):
    """Return a diffrax SDE solver by name (backward compatibility)."""
    import diffrax
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
# NICEFlowSDE — NICE diffusion + path-wise diffrax training / inference
# ---------------------------------------------------------------------------

class NICEFlowSDE(eqx.Module):
    """
    Neural SDE with NICE-structured diffusion integrated via diffrax.

    Physical-time SDE (t ∈ [0, 1]):
        dz_t = f_θ(z_t, c_t, t) dt + G_φ(z_t, c_t, t) dW_t

    dW_t is drawn from VirtualBrownianTree — the latent variable of G_φ.
    G_φ is a NICE coupling network: det(G_φ) = 1 always (no variance collapse).

    Discretised update at each solver step:
        z_{t+dt} = z_t + f_drift(z_t) dt + G_φ(z_t, c_t, t) @ dW_t

    where G_φ(z_t, c_t, t) @ dW_t is the shaped diffusion contribution
    produced by applying the NICE matrix to the Brownian increment dW_t.

    Training modes
    --------------
    pathwise  — integrate SDE with diffrax, minimise ||ẑ_{1:T} − x_{1:T}||²
                (gradients flow back through the solver via the adjoint method)
    stepwise  — per-step NLL via NICE inverse (no integration required):
                0.5 ||G_φ^{-1}(Δx − f dt)||² / dt   (det(G_φ)=1 → no log-det)

    Supported solvers
    -----------------
    "euler"           — Euler-Maruyama       (strong order 0.5)
    "heun"            — Heun                 (strong order 1.0, additive noise)
    "reversible_heun" — ReversibleHeun       (memory-efficient reversible)
    "midpoint"        — Stochastic midpoint
    """
    encoder:    BiGRUEncoder
    drift:      SDEDriftNet
    diffusion:  NICEDiffusionNet
    d_x:        int    = eqx.field(static=True)
    d_y:        int    = eqx.field(static=True)
    physics_fn: object = eqx.field(static=True, default=None)

    def __init__(
        self,
        d_x:           int,
        d_y:           int,
        d_c:           int   = 64,
        d_hidden_enc:  int   = 64,
        d_hidden:      int   = 128,
        depth:         int   = 3,
        time_emb_dim:  int   = 32,
        n_nice_layers: int   = 2,
        physics_fn            = None,
        key:           jr.PRNGKey = jr.PRNGKey(0),
    ):
        self.d_x        = d_x
        self.d_y        = d_y
        self.physics_fn = physics_fn
        k_enc, k_drift, k_diff = jr.split(key, 3)
        self.encoder   = BiGRUEncoder(d_y, d_hidden_enc, d_c, k_enc)
        self.drift     = SDEDriftNet(d_x, d_c, d_hidden, depth, time_emb_dim, k_drift)
        self.diffusion = NICEDiffusionNet(
            d_x, d_c, d_hidden, n_nice_layers, time_emb_dim, k_diff
        )

    def encode(self, y: jnp.ndarray) -> jnp.ndarray:
        """y: (T, d_y) → c: (T, d_c)."""
        return self.encoder(y)

    # ------------------------------------------------------------------
    # Stepwise training:  per-step NLL via NICE inverse
    # ------------------------------------------------------------------

    def _step_nll(
        self,
        z_t:  jnp.ndarray,
        c_t:  jnp.ndarray,
        t:    jnp.ndarray,
        dz:   jnp.ndarray,
        dt:   float,
    ) -> jnp.ndarray:
        """
        EM NLL at one step. det(G_φ) = 1 → no log-det correction.
            0.5 ||G_φ(z_t)^{-1} (Δz − f dt)||² / dt
        """
        f = self.drift(z_t, c_t, t)
        if self.physics_fn is not None:
            f = f + self.physics_fn(z_t)
        resid     = dz - f * dt
        resid_inv = self.diffusion.apply_inverse(resid, z_t, c_t, t)
        return 0.5 * jnp.dot(resid_inv, resid_inv) / dt

    def nll_stepwise_single(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Full-sequence per-step NLL for one (x, y) pair."""
        T  = x.shape[0]
        dt = 1.0 / (T - 1)
        c  = self.encode(y)
        ts = jnp.arange(T - 1) / (T - 1)
        nlls = jax.vmap(
            lambda z, ci, ti, dzi: self._step_nll(z, ci, ti, dzi, dt)
        )(x[:-1], c[:-1], ts, x[1:] - x[:-1])
        return jnp.sum(nlls) / (T - 1)

    def nll_stepwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Batch per-step NLL. x: (B,T,d_x)  y: (B,T,d_y)."""
        return jnp.mean(jax.vmap(self.nll_stepwise_single)(x, y))

    def nll_stepwise_ms_single(
        self,
        x:          jnp.ndarray,
        y:          jnp.ndarray,
        seg_length: int,
    ) -> jnp.ndarray:
        """Multiple-shooting per-step NLL for one trajectory."""
        T      = x.shape[0]
        dt     = 1.0 / (T - 1)
        c      = self.encode(y)
        n_segs = (T - 1) // seg_length
        total  = jnp.array(0.0)
        for k in range(n_segs):
            s     = k * seg_length
            e     = s + seg_length + 1
            x_seg = x[s:e]
            c_seg = c[s:e]
            ts    = (jnp.arange(seg_length) + s) / (T - 1)
            nlls  = jax.vmap(
                lambda z, ci, ti, dzi: self._step_nll(z, ci, ti, dzi, dt)
            )(x_seg[:-1], c_seg[:-1], ts, x_seg[1:] - x_seg[:-1])
            total = total + jnp.sum(nlls) / seg_length
        return total / n_segs

    def nll_stepwise_ms(
        self,
        x:          jnp.ndarray,
        y:          jnp.ndarray,
        seg_length: int,
    ) -> jnp.ndarray:
        """Batch multiple-shooting per-step NLL."""
        return jnp.mean(jax.vmap(
            lambda xi, yi: self.nll_stepwise_ms_single(xi, yi, seg_length)
        )(x, y))

    def diffusion_regularization_single(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Penalise excessive NICE coupling in the diffusion factor.

        The identity part of G_phi is the irreducible Brownian scale, so the
        regularizer targets only G_phi - I. This discourages exaggerated
        uncertainty from the NICE part without asking the diffusion to collapse.
        """
        T = x.shape[0]
        c = self.encode(y)
        ts = jnp.linspace(0.0, 1.0, T)
        eye = jnp.eye(self.d_x)

        def penalty_at(zt, ct, tt):
            G = self.diffusion.matrix(zt, ct, tt)
            return jnp.mean((G - eye) ** 2)

        return jnp.mean(jax.vmap(penalty_at)(x, c, ts))

    def diffusion_regularization(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Batch NICE diffusion coupling penalty."""
        return jnp.mean(jax.vmap(self.diffusion_regularization_single)(x, y))

    # ------------------------------------------------------------------
    # Pathwise training:  integrate SDE → minimise reconstruction loss
    # ------------------------------------------------------------------

    def _build_terms(self, c_interp, diffusion_scale: float = 1.0):
        """
        Build (drift_fn, diff_fn) for use in a diffrax MultiTerm.

        diff_fn returns G_φ(z, c_t, t): shape (d_x, d_x).
        diffrax ControlTerm automatically computes G_φ @ dW_t where
        dW_t is the Brownian increment from the VirtualBrownianTree.
        """

        def drift_fn(t, z, args):
            c_t = c_interp.evaluate(t)
            f   = self.drift(z, c_t, t)
            if self.physics_fn is not None:
                f = f + self.physics_fn(z)
            return f

        def diff_fn(t, z, args):
            c_t = c_interp.evaluate(t)
            return self.diffusion.matrix(z, c_t, t) * diffusion_scale

        return drift_fn, diff_fn

    def _integrate_segment(
        self,
        x_seg:       jnp.ndarray,   # (L+1, d_x)  ground-truth segment
        c_interp,
        t0_seg:      float,
        t1_seg:      float,
        ts_seg:      jnp.ndarray,   # (L+1,)  save times
        key:         jr.PRNGKey,
        dt0:         float,
        solver_name: str,
    ) -> jnp.ndarray:
        """
        Integrate the NICE SDE over one segment, return MSE vs ground truth.

        dW_t ~ VirtualBrownianTree(key)  (the latent noise seed for this segment)
        G_φ(z_t, c_t, t) @ dW_t  —  NICE shapes the Brownian increment.
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
            y0=x_seg[0],
            saveat=diffrax.SaveAt(ts=ts_seg),
        )
        return jnp.mean((sol.ys - x_seg) ** 2)

    def nll_pathwise_single(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        seg_length:       int,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
    ) -> jnp.ndarray:
        """
        Path-wise multiple-shooting reconstruction loss for one trajectory.

        Each segment is independently integrated from its true initial state.
        A fresh VirtualBrownianTree key is drawn per segment; G_φ shapes dW_t.
        Gradients flow back through the solver via the adjoint method.
        
        Args:
            use_custom_solver: If True, use custom stochastic solvers instead of diffrax.
        """
        if use_custom_solver:
            return self._nll_pathwise_single_custom(x, y, key, seg_length, solver_name)
        else:
            return self._nll_pathwise_single_diffrax(x, y, key, seg_length, solver_name)

    def _nll_pathwise_single_diffrax(
        self,
        x:           jnp.ndarray,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        seg_length:  int,
        solver_name: str = "heun",
    ) -> jnp.ndarray:
        """Path-wise loss using diffrax solvers."""
        import diffrax
        T        = x.shape[0]
        dt0      = 1.0 / (T - 1)
        c        = self.encode(y)
        ts_all   = jnp.linspace(0.0, 1.0, T)
        c_interp = diffrax.LinearInterpolation(ts=ts_all, ys=c)

        n_segs   = (T - 1) // seg_length
        seg_keys = jr.split(key, n_segs)
        total    = jnp.array(0.0)
        for k in range(n_segs):
            s      = k * seg_length
            e      = s + seg_length + 1
            t0_seg = s * dt0
            t1_seg = (e - 1) * dt0
            ts_seg = jnp.linspace(t0_seg, t1_seg, seg_length + 1)
            total  = total + self._integrate_segment(
                x[s:e], c_interp, t0_seg, t1_seg, ts_seg,
                seg_keys[k], dt0, solver_name,
            )
        return total / n_segs

    def _nll_pathwise_single_custom(
        self,
        x:           jnp.ndarray,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        seg_length:  int,
        solver_name: str = "heun",
    ) -> jnp.ndarray:
        """Path-wise loss using custom stochastic solvers."""
        T     = x.shape[0]
        dt0   = 1.0 / (T - 1)
        c     = self.encode(y)  # (T, d_c)
        
        n_segs   = (T - 1) // seg_length
        seg_keys = jr.split(key, n_segs)
        total    = jnp.array(0.0)
        
        for k in range(n_segs):
            s      = k * seg_length
            e      = s + seg_length + 1
            t0_seg = s * dt0
            t1_seg = (e - 1) * dt0
            ts_seg = jnp.linspace(t0_seg, t1_seg, seg_length + 1)
            x_pred = self._integrate_custom(
                x[s], ts_seg, c, seg_keys[k], solver_name
            )
            
            # MSE loss
            total = total + jnp.mean((x_pred - x[s:e]) ** 2)
        
        return total / n_segs

    def _context_at(self, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Nearest-neighbour context lookup on the normalised [0, 1] time grid."""
        T = c.shape[0]
        idx = jnp.rint(t * (T - 1)).astype(jnp.int32)
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
        dw:             jnp.ndarray,
        z:              jnp.ndarray,
        c:              jnp.ndarray,
        t:              jnp.ndarray,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        c_t = self._context_at(c, t)
        return self.diffusion.apply(dw * diffusion_temp, z, c_t, t)

    def _custom_solver_step(
        self,
        z:              jnp.ndarray,
        t0:             jnp.ndarray,
        t1:             jnp.ndarray,
        dw:             jnp.ndarray,
        c:              jnp.ndarray,
        solver_name:    str,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """JAX-native fixed-step custom solver update."""
        dt = t1 - t0
        name = solver_name.lower()

        f0 = self._custom_drift(z, c, t0)
        g0 = self._custom_diffusion(dw, z, c, t0, diffusion_temp)

        if name in ("heun", "stochastic_heun", "reversible_heun"):
            z_p = z + f0 * dt + g0
            f1 = self._custom_drift(z_p, c, t1)
            g1 = self._custom_diffusion(dw, z_p, c, t1, diffusion_temp)
            return z + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1)

        if name in ("midpoint", "stochastic_midpoint"):
            t_mid = 0.5 * (t0 + t1)
            z_mid = z + 0.5 * f0 * dt + 0.5 * g0
            f_mid = self._custom_drift(z_mid, c, t_mid)
            g_mid = self._custom_diffusion(dw, z_mid, c, t_mid, diffusion_temp)
            return z + f_mid * dt + g_mid

        if name == "rk4":
            t_mid = 0.5 * (t0 + t1)
            z2 = z + 0.5 * f0 * dt + 0.5 * g0
            f2 = self._custom_drift(z2, c, t_mid)
            g2 = self._custom_diffusion(dw, z2, c, t_mid, diffusion_temp)
            z3 = z + 0.5 * f2 * dt + 0.5 * g2
            f3 = self._custom_drift(z3, c, t_mid)
            g3 = self._custom_diffusion(dw, z3, c, t_mid, diffusion_temp)
            z4 = z + f3 * dt + g3
            f4 = self._custom_drift(z4, c, t1)
            g4 = self._custom_diffusion(dw, z4, c, t1, diffusion_temp)
            return z + (f0 + 2 * f2 + 2 * f3 + f4) * dt / 6.0 + (g0 + 2 * g2 + 2 * g3 + g4) / 6.0

        # Euler-Maruyama fallback, also used for Milstein/adaptive_euler here.
        return z + f0 * dt + g0

    def _integrate_custom(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """Integrate a segment using JAX-native Brownian increments."""
        n_steps = ts.shape[0] - 1
        step_keys = jr.split(key, n_steps)

        def scan_step(z, inputs):
            t0, t1, k = inputs
            dt = t1 - t0
            dw = jr.normal(k, (self.d_x,)) * jnp.sqrt(dt)
            z_next = self._custom_solver_step(
                z, t0, t1, dw, c, solver_name, diffusion_temp
            )
            return z_next, z_next

        _, zs = jax.lax.scan(scan_step, z0, (ts[:-1], ts[1:], step_keys))
        return jnp.concatenate([z0[None], zs], axis=0)

    def nll_pathwise(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        seg_length:       int,
        solver_name:      str = "heun",
        use_custom_solver: bool = False,
    ) -> jnp.ndarray:
        """Batch path-wise reconstruction loss. x: (B,T,d_x)  y: (B,T,d_y)."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.nll_pathwise_single(
                xi, yi, ki, seg_length, solver_name, use_custom_solver
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
        dt_factor:      float = 1.0,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """
        Sample p(x_{1:T} | y_{1:T}) via the NICE SDE.

        Each particle draws a fresh dW_t ~ VirtualBrownianTree.
        G_φ(z_t, c_t, t) @ dW_t shapes the noise into structured diffusion.
        Returns (n_particles, T, d_x).
        """
        import diffrax
        T   = y.shape[0]
        dt0 = dt_factor / (T - 1)
        c   = self.encode(y)
        ts  = jnp.linspace(0.0, 1.0, T)
        c_interp          = diffrax.LinearInterpolation(ts=ts, ys=c)
        drift_fn, diff_fn = self._build_terms(c_interp, diffusion_temp)
        solver            = _get_diffrax_solver(solver_name)
        saveat            = diffrax.SaveAt(ts=ts)

        def simulate_one(k):
            bm = diffrax.VirtualBrownianTree(
                t0=0.0, t1=1.0 + dt0, tol=dt0 * 0.5,
                shape=(self.d_x,), key=k,
            )
            terms = diffrax.MultiTerm(
                diffrax.ODETerm(drift_fn),
                diffrax.ControlTerm(diff_fn, bm),
            )
            sol = diffrax.diffeqsolve(
                terms, solver,
                t0=0.0, t1=1.0, dt0=dt0, y0=x0, saveat=saveat,
            )
            return sol.ys   # (T, d_x)

        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    def sample_posterior_custom_solver(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x0:             jnp.ndarray,   # (d_x,)
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        solver_name:    str   = "heun",
        dt_factor:      float = 1.0,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """
        Sample p(x_{1:T} | y_{1:T}) using custom stochastic solvers.
        
        This method uses our custom SDE solvers (Euler-Maruyama, Heun, Milstein, etc.)
        instead of diffrax, allowing for more control over the integration and
        better integration with the NICE flow diffusion model.
        
        The SDE form is:
            dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, z_t, c_t, t)
        
        where G_φ is the NICE flow that transforms the Brownian increment dW_t.
        
        Args:
            y: Observations (T, d_y)
            x0: Initial state (d_x,)
            key: Random key
            n_particles: Number of particles to sample
            solver_name: Custom solver name ("euler", "heun", "milstein", "midpoint", etc.)
            dt_factor: Time step factor
            diffusion_temp: Temperature scaling for diffusion
        
        Returns:
            Samples (n_particles, T, d_x)
        """
        T   = y.shape[0]
        dt  = dt_factor / (T - 1)
        c   = self.encode(y)
        ts  = jnp.linspace(0.0, 1.0, T)
        
        def simulate_one(particle_key):
            return self._integrate_custom(
                x0, ts, c, particle_key, solver_name, diffusion_temp
            )
        
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

    def sample_posterior_gt_segmented(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x:              jnp.ndarray,   # (T, d_x)  ground-truth
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        seg_length:     int   = 20,
        solver_name:    str   = "heun",
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """
        Ground-truth-seeded segmented posterior sampling.

        Each segment is reinitialised from x[k*seg_length], preventing
        exponential error growth over chaotic horizons.
        Returns (n_particles, T, d_x).
        """
        import diffrax
        T      = y.shape[0]
        dt0    = 1.0 / (T - 1)
        c      = self.encode(y)
        n_segs = (T + seg_length - 1) // seg_length
        ts_all = jnp.linspace(0.0, 1.0, T)
        c_interp          = diffrax.LinearInterpolation(ts=ts_all, ys=c)
        drift_fn, diff_fn = self._build_terms(c_interp, diffusion_temp)
        solver            = _get_diffrax_solver(solver_name)

        def simulate_segment(k_seg, p_key):
            start  = k_seg * seg_length
            end    = min(start + seg_length, T)
            L      = end - start
            if L <= 1:
                return x[start][None]
            t0_seg = start * dt0
            t1_seg = (end - 1) * dt0
            ts_seg = jnp.linspace(t0_seg, t1_seg, L)
            bm = diffrax.VirtualBrownianTree(
                t0=t0_seg, t1=t1_seg + dt0, tol=dt0 * 0.5,
                shape=(self.d_x,), key=p_key,
            )
            terms = diffrax.MultiTerm(
                diffrax.ODETerm(drift_fn),
                diffrax.ControlTerm(diff_fn, bm),
            )
            sol = diffrax.diffeqsolve(
                terms, solver,
                t0=t0_seg, t1=t1_seg, dt0=dt0,
                y0=x[start],
                saveat=diffrax.SaveAt(ts=ts_seg),
            )
            return sol.ys   # (L, d_x)

        def run_one_particle(p_key):
            seg_keys = jr.split(p_key, n_segs)
            segs     = [simulate_segment(k, seg_keys[k]) for k in range(n_segs)]
            return jnp.concatenate(segs, axis=0)[:T]   # (T, d_x)

        return jax.vmap(run_one_particle)(jr.split(key, n_particles))


# ---------------------------------------------------------------------------
# NICEFlowSDE training utilities
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step_nice_stepwise(
    model:     NICEFlowSDE,
    x_batch:   jnp.ndarray,
    y_batch:   jnp.ndarray,
    opt_state,
    optimizer,
    diffusion_reg_weight: float = 0.0,
):
    """Per-step NLL training (NICE inverse, no SDE integration)."""
    loss_fn = lambda m: (
        m.nll_stepwise(x_batch, y_batch)
        + diffusion_reg_weight * m.diffusion_regularization(x_batch, y_batch)
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss


@eqx.filter_jit
def make_step_nice_stepwise_ms(
    model:      NICEFlowSDE,
    x_batch:    jnp.ndarray,
    y_batch:    jnp.ndarray,
    seg_length: int,
    opt_state,
    optimizer,
    diffusion_reg_weight: float = 0.0,
):
    """Multiple-shooting per-step NLL training."""
    loss_fn = lambda m: (
        m.nll_stepwise_ms(x_batch, y_batch, seg_length)
        + diffusion_reg_weight * m.diffusion_regularization(x_batch, y_batch)
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss


@eqx.filter_jit
def make_step_nice_pathwise(
    model:            NICEFlowSDE,
    x_batch:          jnp.ndarray,
    y_batch:          jnp.ndarray,
    key:              jr.PRNGKey,
    seg_length:       int,
    opt_state,
    optimizer,
    solver_name:      str = "heun",
    use_custom_solver: bool = False,
    diffusion_reg_weight: float = 0.0,
):
    """
    Path-wise training step.

    Integrates the NICE SDE for each multiple-shooting segment.
    dW_t ~ VirtualBrownianTree (one independent tree per segment per sample).
    G_φ(z_t, c_t, t) @ dW_t shapes the noise; gradients flow through the solver.
    
    Args:
        use_custom_solver: If True, use custom stochastic solvers from stochastic_solvers.py
                          instead of diffrax. Supports: euler, heun, reversible_heun, 
                          midpoint, milstein, adaptive_euler, rk4.
    """
    loss_fn = lambda m: (
        m.nll_pathwise(
            x_batch, y_batch, key, seg_length, solver_name, use_custom_solver
        )
        + diffusion_reg_weight * m.diffusion_regularization(x_batch, y_batch)
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss
