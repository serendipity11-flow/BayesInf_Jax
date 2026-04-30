"""
NICEFlowSDE: Flow-SDE for posterior trajectory estimation.

Model:
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, c_t, t) \sqrt(dt)

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

Training:
    Pathwise multiple-shooting reconstruction loss. The custom solver path uses
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, z_t, c_t, t) sqrt(dt).

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
# NICEFlowSDE — NICE diffusion + path-wise diffrax training / inference
# ---------------------------------------------------------------------------

class NICEFlowSDE(eqx.Module):
    """
    Neural SDE with NICE-structured diffusion integrated via diffrax.

    Physical-time SDE (t ∈ [0, 1]):
        dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, c_t, t) \sqrt(dt)

    dW_t is drawn from VirtualBrownianTree — the latent variable of G_φ.
    G_φ is a NICE coupling network: det(G_φ) = 1 always (no variance collapse).

    Discretised update at each solver step:
        z_{t+dt} = z_t + f_drift(z_t) dt + G_φ(dW_t, c_t, t) \sqrt(dt)

    where G_φ(dW_t, c_t, t) \sqrt(dt) is the shaped diffusion contribution
    produced by applying the NICE matrix to the Brownian increment dW_t.

    Training
    --------
    pathwise — integrate trajectories and minimise ||ẑ_{1:T} − x_{1:T}||².
               With use_custom_solver=True, the update convention is
               dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, z_t, c_t, t) sqrt(dt).

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

    def _integrate_custom(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """Integrate a segment using local solvers through diffrax.diffeqsolve."""
        import diffrax

        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0]
        c_interp = diffrax.LinearInterpolation(ts=ts, ys=c)
        drift_fn, diff_fn = self._build_terms(c_interp, diffusion_temp)
        solver = _get_diffrax_solver(solver_name)
        bm = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1 + dt0, tol=dt0 * 0.5,
            shape=(self.d_x,), key=key,
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn),
            diffrax.ControlTerm(diff_fn, bm),
        )
        sol = diffrax.diffeqsolve(
            terms, solver,
            t0=t0, t1=t1, dt0=dt0, y0=z0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys
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
        
        The custom solver form is:
            dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, z_t, c_t, t) sqrt(dt)

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
        use_custom_solver: bool = False,
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
        n_segs = (T - 1) // seg_length
        ts_all = jnp.linspace(0.0, 1.0, T)

        if use_custom_solver:
            def simulate_segment(k_seg, p_key):
                start  = k_seg * seg_length
                end    = start + seg_length + 1
                t0_seg = start * dt0
                t1_seg = (end - 1) * dt0
                ts_seg = jnp.linspace(t0_seg, t1_seg, seg_length + 1)
                return self._integrate_custom(
                    x[start], ts_seg, c, p_key, solver_name, diffusion_temp
                )

            def run_one_particle(p_key):
                seg_keys = jr.split(p_key, n_segs)
                segs     = [simulate_segment(k, seg_keys[k]) for k in range(n_segs)]
                pieces   = [segs[0]] + [seg[1:] for seg in segs[1:]]
                return jnp.concatenate(pieces, axis=0)[:T]

            return jax.vmap(run_one_particle)(jr.split(key, n_particles))

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
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss
