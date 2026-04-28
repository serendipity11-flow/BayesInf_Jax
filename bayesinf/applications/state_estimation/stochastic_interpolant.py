"""
StochasticInterpolantSE — Signal-to-Signal Stochastic Interpolant for State Estimation.

Default interpolant (linear path):
    z_s = (1-s)·z_0 + s·x + σ_s·ε
    σ_s = σ_0 · s · (1-s)

    α(s) = 1-s,   α'(s) = -1
    β(s) = s,     β'(s) =  1
    σ̇(s) = σ_0 · (1 - 2s)

Source z_0 (per particle, per time-step):
    observed dims   : z_0 = ỹ = Lifter(y, c)   — tight, deterministic source
    unobserved dims : z_0 ~ N(0, 1)             — Gaussian prior, diverse per particle

Velocity and score networks both condition on ỹ (the lifter output) explicitly,
which lets them directly compute deviations from z_0 without inferring the source.

Refinement SDE (flow time s ∈ [0, 1]):
    dz_s = [v_θ(z_s, ỹ, c, s) + g₀² · score_ψ(z_s, ỹ, c, s)] ds
           + g₀ · √(2σ_s) dW_s

Training — joint flow + σ-scaled score matching:
    z_0    = masked_source(ỹ, ε_0, obs_mask)
    z_s    = α(s)·z_0 + β(s)·x + σ_s·ε
    v_tgt  = α'(s)·z_0 + β'(s)·x + σ̇_s·ε
    L = ||v_θ - v_tgt||² + λ·||score_ψ + ε||² + μ·||ỹ - x||²
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

from bayesinf.applications.state_estimation.flow_sde import (
    SinusoidalEmb,
)


# ---------------------------------------------------------------------------
# Observation encoder:  (T, d_y) → (T, d_c)   [transformer, global attention]
# ---------------------------------------------------------------------------

class ObsTransformerEncoder(eqx.Module):
    """
    Transformer encoder: y (L, d_y) → c (L, d_c).

    Processes the full observation sequence with bidirectional self-attention
    so each time step's context vector sees all other observations.
    Replaces the causal BiGRU encoder.
    """
    step_emb:    SinusoidalEmb
    input_proj:  eqx.nn.Linear
    layers:      list
    output_proj: eqx.nn.Linear

    def __init__(self, d_y: int, d_c: int, d_model: int,
                 n_heads: int, depth: int, time_emb_dim: int, key: jr.PRNGKey):
        k_in, k_out, *k_layers = jr.split(key, depth + 2)
        in_size = d_y + time_emb_dim
        self.step_emb    = SinusoidalEmb(time_emb_dim)
        self.input_proj  = eqx.nn.Linear(in_size, d_model, key=k_in)
        self.layers      = [TransformerBlock(d_model, n_heads, k_layers[i])
                            for i in range(depth)]
        self.output_proj = eqx.nn.Linear(d_model, d_c, key=k_out)

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """y: (L, d_y) → c: (L, d_c)."""
        L = y.shape[0]
        step_norm = jnp.arange(L, dtype=jnp.float32) / jnp.maximum(L - 1, 1)
        p_seq = jax.vmap(self.step_emb)(step_norm)
        x = jnp.concatenate([y, p_seq], axis=-1)
        x = jax.vmap(self.input_proj)(x)
        for layer in self.layers:
            x = layer(x)
        return jax.vmap(self.output_proj)(x)


# ---------------------------------------------------------------------------
# Observation lifter:  (T, d_y) + (T, d_c) → (T, d_x)   [per-step MLP]
# ---------------------------------------------------------------------------

class StateLifterMLP(eqx.Module):
    """
    Per-step MLP lifter: (y_t, c_t) → ỹ_t, applied independently at each step.

    Simple and fast — the encoder transformer already provides global context
    in c_t, so the lifter only needs to project into state space.
    """
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(self, d_y: int, d_x: int, d_c: int, d_hidden: int, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(d_y + d_c, d_hidden, key=k1)
        self.fc2 = eqx.nn.Linear(d_hidden,  d_hidden,  key=k2)
        self.fc3 = eqx.nn.Linear(d_hidden,  d_x,       key=k3)

    def _step(self, y_t: jnp.ndarray, c_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([y_t, c_t], axis=-1)
        x = jax.nn.gelu(self.fc1(x))
        x = jax.nn.gelu(self.fc2(x))
        return self.fc3(x)

    def __call__(self, y: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """y: (L, d_y)  c: (L, d_c) → (L, d_x)."""
        return jax.vmap(self._step)(y, c)


# ---------------------------------------------------------------------------
# Transformer velocity / score networks
# ---------------------------------------------------------------------------

class TransformerBlock(eqx.Module):
    """Pre-norm transformer block: self-attention + FFN."""
    attn:  eqx.nn.MultiheadAttention
    ff1:   eqx.nn.Linear
    ff2:   eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.attn  = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.ff1   = eqx.nn.Linear(d_model, 4 * d_model, key=k2)
        self.ff2   = eqx.nn.Linear(4 * d_model, d_model, key=k3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (L, d_model) → (L, d_model)."""
        n1 = jax.vmap(self.norm1)(x)
        x  = x + self.attn(n1, n1, n1)
        n2 = jax.vmap(self.norm2)(x)
        h  = jax.vmap(lambda xi: self.ff2(jax.nn.gelu(self.ff1(xi))))(n2)
        return x + h


class SITransformerNet(eqx.Module):
    """
    Transformer velocity / score network for StochasticInterpolantSE.

    Processes all L time steps jointly via bidirectional self-attention,
    capturing temporal dependencies within each segment.
    Input per step: z_s ‖ ỹ ‖ c ‖ flow_emb(s) ‖ pos_emb(step/L)
    """
    time_emb:    SinusoidalEmb
    step_emb:    SinusoidalEmb
    input_proj:  eqx.nn.Linear
    layers:      list
    output_proj: eqx.nn.Linear

    def __init__(self, d_x: int, d_c: int, d_model: int, n_heads: int,
                 depth: int, time_emb_dim: int, key: jr.PRNGKey):
        k_t, k_p, k_in, k_out, *k_layers = jr.split(key, depth + 4)
        in_size = 2 * d_x + d_c + 2 * time_emb_dim
        self.time_emb    = SinusoidalEmb(time_emb_dim)
        self.step_emb    = SinusoidalEmb(time_emb_dim)
        self.input_proj  = eqx.nn.Linear(in_size, d_model, key=k_in)
        self.layers      = [TransformerBlock(d_model, n_heads, k_layers[i])
                            for i in range(depth)]
        self.output_proj = eqx.nn.Linear(d_model, d_x, key=k_out)

    def __call__(self, z_s: jnp.ndarray, y_lift: jnp.ndarray,
                 c: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        """z_s: (L, d_x)  y_lift: (L, d_x)  c: (L, d_c)  s: scalar → (L, d_x)."""
        L     = z_s.shape[0]
        t_emb = self.time_emb(s)
        t_seq = jnp.broadcast_to(t_emb, (L, t_emb.shape[0]))

        step_norm = jnp.arange(L, dtype=jnp.float32) / jnp.maximum(L - 1, 1)
        p_seq = jax.vmap(self.step_emb)(step_norm)

        x = jnp.concatenate([z_s, y_lift, c, t_seq, p_seq], axis=-1)
        x = jax.vmap(self.input_proj)(x)
        for layer in self.layers:
            x = layer(x)
        return jax.vmap(self.output_proj)(x)

    def traj(self, z_s, y_lift, c, s):
        return self(z_s, y_lift, c, s)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StochasticInterpolantSE(eqx.Module):
    """
    Signal-to-Signal Stochastic Interpolant for State Estimation.

    Source distribution:
      - Observed dims   : z_0 = ỹ = Lifter(y, c)   (deterministic, tight)
      - Unobserved dims : z_0 ~ N(0, 1)              (stochastic, one draw per particle)
    Each particle draws its own z_0^unobs, giving posterior diversity that does not
    depend on SDE noise accumulation over the long chaotic integration horizon.
    """
    encoder:            ObsTransformerEncoder
    lifter:             StateLifterMLP
    vel_net:            SITransformerNet
    score_net:          SITransformerNet
    d_x:                int   = eqx.field(static=True)
    d_y:                int   = eqx.field(static=True)
    d_c:                int   = eqx.field(static=True)
    obs_interval:       int   = eqx.field(static=True)
    sigma_0:            float = eqx.field(static=True)
    g0:                 float = eqx.field(static=True)
    score_weight:       float = eqx.field(static=True)
    lifter_loss_weight: float = eqx.field(static=True)
    s_min:              float = eqx.field(static=True)
    interpolant_type:   str   = eqx.field(static=True)

    def __init__(
        self,
        d_x:                int,
        d_y:                int,
        d_c:                int   = 64,
        d_hidden:           int   = 128,
        n_heads:            int   = 4,
        depth:              int   = 3,
        time_emb_dim:       int   = 32,
        obs_interval:       int   = 2,
        sigma_0:            float = 1.0,
        g0:                 float = 0.5,
        score_weight:       float = 1.0,
        lifter_loss_weight: float = 0.5,
        s_min:              float = 0.02,
        interpolant_type:   str   = "linear",
        key:                jr.PRNGKey = jr.PRNGKey(0),
    ):
        if interpolant_type not in ("cosine", "linear"):
            raise ValueError(f"interpolant_type must be 'cosine' or 'linear', got {interpolant_type}")
        self.d_x            = d_x
        self.d_y            = d_y
        self.d_c            = d_c
        self.obs_interval   = obs_interval
        self.sigma_0        = sigma_0
        self.g0             = g0
        self.score_weight   = score_weight
        self.lifter_loss_weight = lifter_loss_weight
        self.s_min          = s_min
        self.interpolant_type = interpolant_type

        k_enc, k_lift, k_vel, k_score = jr.split(key, 4)
        self.encoder   = ObsTransformerEncoder(d_y, d_c, d_hidden, n_heads, depth, time_emb_dim, k_enc)
        self.lifter    = StateLifterMLP(d_y, d_x, d_c, d_hidden, k_lift)
        self.vel_net   = SITransformerNet(d_x, d_c, d_hidden, n_heads, depth, time_emb_dim, k_vel)
        self.score_net = SITransformerNet(d_x, d_c, d_hidden, n_heads, depth, time_emb_dim, k_score)

    # ------------------------------------------------------------------
    # σ schedule
    # ------------------------------------------------------------------

    # Cosine interpolant coefficients
    def _alpha_cosine(self, s):
        return jnp.cos(jnp.pi / 2.0 * s)

    def _beta_cosine(self, s):
        return jnp.sin(jnp.pi / 2.0 * s)

    def _alpha_dot_cosine(self, s):
        return -jnp.pi / 2.0 * jnp.sin(jnp.pi / 2.0 * s)

    def _beta_dot_cosine(self, s):
        return jnp.pi / 2.0 * jnp.cos(jnp.pi / 2.0 * s)

    def _sigma_cosine(self, s):
        # σ_0 · α(s) · β(s) = σ_0/2 · sin(πs) — zero at both endpoints
        return self.sigma_0 * self._alpha_cosine(s) * self._beta_cosine(s)

    def _sigma_dot_cosine(self, s):
        # d/ds [σ_0 · cos(π/2·s) · sin(π/2·s)] = σ_0 · π/2 · cos(πs)
        return self.sigma_0 * jnp.pi / 2.0 * jnp.cos(jnp.pi * s)

    # Linear interpolant coefficients
    def _alpha_linear(self, s):
        return 1.0 - s

    def _beta_linear(self, s):
        return s

    def _alpha_dot_linear(self, s):
        return -jnp.ones_like(s)

    def _beta_dot_linear(self, s):
        return jnp.ones_like(s)

    def _sigma_linear(self, s):
        # σ_0 · (1-s) · s = σ_0 · s(1-s) — zero at both endpoints
        return self.sigma_0 * (1.0 - s) * s

    def _sigma_dot_linear(self, s):
        # d/ds [σ_0 · s(1-s)] = σ_0 · (1 - 2s)
        return self.sigma_0 * (1.0 - 2.0 * s)

    # Unified interface based on interpolant_type
    def _alpha(self, s):
        if self.interpolant_type == "cosine":
            return self._alpha_cosine(s)
        else:  # linear
            return self._alpha_linear(s)

    def _beta(self, s):
        if self.interpolant_type == "cosine":
            return self._beta_cosine(s)
        else:  # linear
            return self._beta_linear(s)

    def _alpha_dot(self, s):
        if self.interpolant_type == "cosine":
            return self._alpha_dot_cosine(s)
        else:  # linear
            return self._alpha_dot_linear(s)

    def _beta_dot(self, s):
        if self.interpolant_type == "cosine":
            return self._beta_dot_cosine(s)
        else:  # linear
            return self._beta_dot_linear(s)

    def _sigma(self, s):
        if self.interpolant_type == "cosine":
            return self._sigma_cosine(s)
        else:  # linear
            return self._sigma_linear(s)

    def _sigma_dot(self, s):
        if self.interpolant_type == "cosine":
            return self._sigma_dot_cosine(s)
        else:  # linear
            return self._sigma_dot_linear(s)

    # ------------------------------------------------------------------
    # Observed-dimension mask  (static, derived from d_x and obs_interval)
    # ------------------------------------------------------------------

    def _obs_mask(self) -> jnp.ndarray:
        """1 at observed dims, 0 at unobserved dims. Constant at JIT time."""
        idx = jnp.arange(0, self.d_x, self.obs_interval)
        return jnp.zeros(self.d_x).at[idx].set(1.0)

    # ------------------------------------------------------------------
    # Masked source: observed dims from ỹ, unobserved dims from N(0,1)
    # ------------------------------------------------------------------

    def _make_source(
        self,
        y_lift: jnp.ndarray,   # (L, d_x)  lifter output
        key:    jr.PRNGKey,
    ) -> jnp.ndarray:          # (L, d_x)  z_0 with stochastic unobs dims
        obs_mask = self._obs_mask()           # (d_x,)  broadcast over L
        noise    = jr.normal(key, y_lift.shape)
        return y_lift * obs_mask + noise * (1.0 - obs_mask)

    # ------------------------------------------------------------------
    # Core SDE simulation primitive  (one particle, one segment)
    # ------------------------------------------------------------------

    def _simulate_sde_segment(
        self,
        z0:         jnp.ndarray,   # (L, d_x)  initial trajectory (with source noise)
        y_lift_seg: jnp.ndarray,   # (L, d_x)  lifter output — conditioning for nets
        c_seg:      jnp.ndarray,   # (L, d_c)  BiGRU context
        key:        jr.PRNGKey,
        n_steps:    int,
        g0:         float,
    ) -> jnp.ndarray:              # (L, d_x)  refined trajectory
        """
        Integrate the Refinement SDE from s=0 to s=1 for one particle.
        y_lift_seg is passed separately so the networks always see the same
        conditioning signal regardless of the stochastic z_0.
        """
        ds        = 1.0 / n_steps
        s_vals    = jnp.linspace(0.0, 1.0 - ds, n_steps)
        step_keys = jr.split(key, n_steps)

        def em_step(z, inputs):
            s_i, noise_k = inputs
            v         = self.vel_net.traj(z, y_lift_seg, c_seg, s_i)
            score     = self.score_net.traj(z, y_lift_seg, c_seg, s_i)
            drift     = v + g0 ** 2 * score
            diff_coef = g0 * jnp.sqrt(2.0 * self._sigma(s_i) + 1e-8)
            noise     = diff_coef * jr.normal(noise_k, z.shape) * jnp.sqrt(ds)
            return z + drift * ds + noise, None

        z1, _ = jax.lax.scan(em_step, z0, (s_vals, step_keys))
        return z1

    # ------------------------------------------------------------------
    # Core loss primitive  (one segment, pre-computed context + lift)
    # ------------------------------------------------------------------

    def _loss_seg(
        self,
        x_seg:      jnp.ndarray,   # (L, d_x)  ground-truth state segment
        y_lift_seg: jnp.ndarray,   # (L, d_x)  lifted observation segment
        c_seg:      jnp.ndarray,   # (L, d_c)  encoded context segment
        key:        jr.PRNGKey,
    ) -> jnp.ndarray:
        """
        Joint flow + σ-scaled score loss.

        Source z_0: observed dims from ỹ, unobserved dims from N(0,1).
        Velocity target: (x - z_0) + σ_dot * ε  — depends on z_0, hence on key.
        Score target: -ε  — independent of z_0.
        """
        k_s, k_eps, k_z0 = jr.split(key, 3)
        s   = jr.uniform(k_s, minval=self.s_min, maxval=1.0 - self.s_min)
        eps = jr.normal(k_eps, x_seg.shape)

        # Mixed source: tight for observed dims, diverse Gaussian for unobserved
        z0_seg = self._make_source(y_lift_seg, k_z0)   # (L, d_x)

        sig_s  = self._sigma(s)
        z_s    = self._alpha(s) * z0_seg + self._beta(s) * x_seg + sig_s * eps

        v_target   = self._alpha_dot(s) * z0_seg + self._beta_dot(s) * x_seg + self._sigma_dot(s) * eps
        v_pred     = self.vel_net.traj(z_s, y_lift_seg, c_seg, s)
        l_flow     = jnp.mean((v_pred - v_target) ** 2)

        score_pred = self.score_net.traj(z_s, y_lift_seg, c_seg, s)
        l_score    = jnp.mean((score_pred + eps) ** 2)

        l_lift = jnp.mean((y_lift_seg - x_seg) ** 2)

        return l_flow + self.score_weight * l_score + self.lifter_loss_weight * l_lift

    # ------------------------------------------------------------------
    # Full-trajectory training loss
    # ------------------------------------------------------------------

    def loss_single(self, x, y, key):
        """Full-trajectory loss for one (x, y) pair."""
        c      = self.encoder(y)
        y_lift = self.lifter(y, c)
        return self._loss_seg(x, y_lift, c, key)

    def loss(self, x, y, key):
        """Batch full-trajectory loss. x: (B,T,d_x)  y: (B,T,d_y)."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(self.loss_single)(x, y, keys))

    # ------------------------------------------------------------------
    # Multiple-shooting training loss
    # ------------------------------------------------------------------

    def loss_ms_single(
        self,
        x:       jnp.ndarray,   # (T, d_x)
        y:       jnp.ndarray,   # (T, d_y)
        key:     jr.PRNGKey,
        seg_len: int,
    ) -> jnp.ndarray:
        T      = x.shape[0]
        n_segs = (T - 1) // seg_len
        L      = seg_len + 1

        c_full      = self.encoder(y)
        y_lift_full = self.lifter(y, c_full)

        seg_starts = (jnp.arange(n_segs) * seg_len).astype(jnp.int32)

        x_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(x,          (s, 0), (L, self.d_x)))(seg_starts)
        y_lift_segs = jax.vmap(lambda s: jax.lax.dynamic_slice(y_lift_full, (s, 0), (L, self.d_x)))(seg_starts)
        c_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(c_full,      (s, 0), (L, self.d_c)))(seg_starts)

        keys   = jr.split(key, n_segs)
        losses = jax.vmap(self._loss_seg)(x_segs, y_lift_segs, c_segs, keys)
        return jnp.mean(losses)

    def loss_ms_batch(self, x, y, key, seg_len):
        """Batch multiple-shooting loss."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.loss_ms_single(xi, yi, ki, seg_len)
        )(x, y, keys))

    # ------------------------------------------------------------------
    # Full-trajectory SDE sampler
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        n_particles: int   = 100,
        n_steps:     int   = 100,
        g0_override: float = None,
    ) -> jnp.ndarray:
        """
        Sample from p(x_{1:T} | y_{1:T}) via the full-trajectory Refinement SDE.
        Each particle draws its own z_0^unobs ~ N(0,1) → diverse posterior.
        Returns (n_particles, T, d_x).
        """
        c      = self.encoder(y)
        y_lift = self.lifter(y, c)
        g0     = self.g0 if g0_override is None else g0_override

        def simulate_one(pk):
            k_z0, k_sde = jr.split(pk)
            z0 = self._make_source(y_lift, k_z0)   # per-particle stochastic source
            return self._simulate_sde_segment(z0, y_lift, c, k_sde, n_steps, g0)

        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    # ------------------------------------------------------------------
    # Deterministic ODE sampler
    # ------------------------------------------------------------------

    def sample_posterior_ode(
        self,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        n_particles: int = 100,
        n_steps:     int = 100,
    ) -> jnp.ndarray:
        """
        Probability-flow ODE sampler.
        Each particle draws its own z_0^unobs for diversity.
        Returns (n_particles, T, d_x).
        """
        ds     = 1.0 / n_steps
        c      = self.encoder(y)
        y_lift = self.lifter(y, c)
        s_vals = jnp.linspace(0.0, 1.0 - ds, n_steps)

        def simulate_one(pk):
            z0 = self._make_source(y_lift, pk)
            def euler_step(z, s_i):
                v = self.vel_net.traj(z, y_lift, c, s_i)
                return z + v * ds, None
            z1, _ = jax.lax.scan(euler_step, z0, s_vals)
            return z1

        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    # ------------------------------------------------------------------
    # Segmented SDE sampler  (boundary-pinned, lax.scan over segments)
    # ------------------------------------------------------------------

    def sample_posterior_segmented(
        self,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        n_particles: int   = 100,
        n_steps:     int   = 100,
        seg_len:     int   = 20,
        g0_override: float = None,
    ) -> jnp.ndarray:
        """
        Segmented posterior sampling with boundary pinning.

        At each segment k, each particle draws its own z_0^unobs ~ N(0,1)
        for the interior time steps, giving diverse posterior trajectories.
        The boundary (t=0 of each segment) carries the last refined state
        from segment k-1.

        Returns (n_particles, n_segs*seg_len+1, d_x).
        """
        T      = y.shape[0]
        n_segs = (T - 1) // seg_len
        L      = seg_len + 1
        g0     = self.g0 if g0_override is None else g0_override

        c_full      = self.encoder(y)
        y_lift_full = self.lifter(y, c_full)

        obs_mask = self._obs_mask()   # (d_x,) — constant

        def process_segment(boundary, k):
            """
            boundary : (n_particles, d_x) — last refined state of prev segment
            k        : scalar segment index
            """
            start      = k * seg_len
            c_seg      = jax.lax.dynamic_slice(c_full,      (start, 0), (L, self.d_c))
            y_lift_seg = jax.lax.dynamic_slice(y_lift_full, (start, 0), (L, self.d_x))

            p_keys = jr.split(jr.fold_in(key, k), n_particles)

            def sim_one(bnd, pk):
                k_z0, k_sde = jr.split(pk)
                # Interior steps t=1..L-1: stochastic source for unobs dims
                noise_rest  = jr.normal(k_z0, (seg_len, self.d_x))
                z0_rest     = y_lift_seg[1:] * obs_mask + noise_rest * (1.0 - obs_mask)
                # First step t=0: boundary from previous segment (carries refined state)
                z0_all      = jnp.concatenate([bnd[None], z0_rest], axis=0)  # (L, d_x)
                return self._simulate_sde_segment(z0_all, y_lift_seg, c_seg, k_sde, n_steps, g0)

            seg_out      = jax.vmap(sim_one)(boundary, p_keys)   # (n_p, L, d_x)
            new_boundary = seg_out[:, -1, :]                      # (n_p, d_x)
            return new_boundary, seg_out

        # Segment 0: first time-step boundary is ỹ[0] for obs dims, N(0,1) for unobs,
        # independently per particle.
        k_bnd0, key = jr.split(key)
        bnd0_noise  = jr.normal(k_bnd0, (n_particles, self.d_x))
        boundary_0  = (jnp.broadcast_to(y_lift_full[0], (n_particles, self.d_x)) * obs_mask
                       + bnd0_noise * (1.0 - obs_mask))

        _, all_seg_outs = jax.lax.scan(
            process_segment,
            boundary_0,
            jnp.arange(n_segs, dtype=jnp.int32),
        )  # (n_segs, n_p, L, d_x)

        seg0 = all_seg_outs[0]                          # (n_p, L, d_x)
        rest = all_seg_outs[1:, :, 1:, :]              # (n_segs-1, n_p, seg_len, d_x)
        rest = jnp.swapaxes(rest, 0, 1)                 # (n_p, n_segs-1, seg_len, d_x)
        rest = rest.reshape(n_particles, (n_segs - 1) * seg_len, self.d_x)
        return jnp.concatenate([seg0, rest], axis=1)    # (n_p, n_segs*seg_len+1, d_x)

    # ------------------------------------------------------------------
    # Batch wrappers
    # ------------------------------------------------------------------

    def sample_posterior_batch(
        self,
        y:           jnp.ndarray,   # (B, T, d_y)
        key:         jr.PRNGKey,
        n_particles: int  = 50,
        n_steps:     int  = 100,
        stochastic:  bool = True,
        segmented:   bool = False,
        seg_len:     int  = 20,
    ) -> jnp.ndarray:
        """Returns (n_particles, B, T', d_x)."""
        B    = y.shape[0]
        keys = jr.split(key, B)

        if segmented:
            fn = lambda yi, ki: self.sample_posterior_segmented(
                yi, ki, n_particles, n_steps, seg_len
            )
        elif stochastic:
            fn = lambda yi, ki: self.sample_posterior(yi, ki, n_particles, n_steps)
        else:
            fn = lambda yi, ki: self.sample_posterior_ode(yi, ki, n_particles, n_steps)

        samples = jax.vmap(fn)(y, keys)         # (B, n_p, T', d_x)
        return jnp.swapaxes(samples, 0, 1)      # (n_p, B, T', d_x)


# ---------------------------------------------------------------------------
# JIT-compiled training steps
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step_si(
    model:     StochasticInterpolantSE,
    x_batch:   jnp.ndarray,
    y_batch:   jnp.ndarray,
    key:       jr.PRNGKey,
    opt_state,
    optimizer,
):
    """Full-trajectory training step."""
    loss_fn = lambda m: m.loss(x_batch, y_batch, key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss


@eqx.filter_jit
def make_step_si_ms(
    model:     StochasticInterpolantSE,
    x_batch:   jnp.ndarray,
    y_batch:   jnp.ndarray,
    key:       jr.PRNGKey,
    seg_len:   int,
    opt_state,
    optimizer,
):
    """Multiple-shooting training step."""
    loss_fn = lambda m: m.loss_ms_batch(x_batch, y_batch, key, seg_len)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss
