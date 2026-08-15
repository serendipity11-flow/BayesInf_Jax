"""
StochasticInterpolantSE — Signal-to-Signal Stochastic Interpolant for State Estimation.

Default interpolant (linear path):
    z_s = (1-s)·z_0 + s·x + σ_s·ε
    σ_s = σ_0 · s · (1-s)

    α(s) = 1-s,   α'(s) = -1
    β(s) = s,     β'(s) =  1
    σ̇(s) = σ_0 · (1 - 2s)

Source z_0 (per particle, per time-step), selected by ``source_type``:
    inferred: q_η(y_{1:T}) = N(μ_η(y), diag(σ_η(y)^2))
    gaussian: z_0 ~ N(base_mean, base_std² I)

For an inferred source, velocity and score condition on (μ_η, c). For a
fixed Gaussian source there is no inference network and both fields condition
only on the encoded observation context c.

Refinement SDE (flow time s ∈ [0, 1]):
    dz_s = [v_θ(z_s, μ_η, c, s) + g₀² · score_ψ(z_s, μ_η, c, s)] ds
           + g₀ · √(2σ_s) dW_s

Training — joint flow + σ-scaled score matching:
    z_0    = μ_η + σ_η · ε_0
    z_s    = α(s)·z_0 + β(s)·x + σ_s·ε
    v_tgt  = α'(s)·z_0 + β'(s)·x + σ̇_s·ε
    L = ||v_θ - v_tgt||² + λ·||score_ψ + ε||²
        + μ·NLL_qη(x | y)
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


def linear_interpolant_coefficients(s, sigma_0):
    """Return coefficients and derivatives for the shared linear SI path."""
    alpha = 1.0 - s
    beta = s
    sigma = sigma_0 * s * (1.0 - s)
    return (
        alpha, beta, sigma, -jnp.ones_like(s), jnp.ones_like(s),
        sigma_0 * (1.0 - 2.0 * s),
    )


def unscale_interpolant_score(scaled_score, sigma, sigma_floor):
    """Convert a stable ``-epsilon`` prediction into an actual score."""
    return scaled_score / jnp.maximum(sigma, sigma_floor)


def refinement_sde_coefficients(sigma, g0_squared):
    """Return score-drift weight and diffusion for the refinement SDE."""
    return g0_squared, jnp.sqrt(2.0 * g0_squared * sigma + 1e-8)


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
# Inference network:  (T, d_y) → (T, d_x), (T, d_x)
# ---------------------------------------------------------------------------

class StateInferenceMLP(eqx.Module):
    """
    Per-step inference network q_eta(z_0 | y_{1:T}).

    The encoder transformer provides global observation context in c_t; this
    head returns diagonal-Gaussian source parameters in state space.
    """
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    mu_head: eqx.nn.Linear
    log_sigma_head: eqx.nn.Linear

    def __init__(self, d_y: int, d_x: int, d_hidden: int, key: jr.PRNGKey):
        k1, k2, k_mu, k_sig = jr.split(key, 4)
        self.fc1 = eqx.nn.Linear(d_y, d_hidden, key=k1)
        self.fc2 = eqx.nn.Linear(d_hidden,  d_hidden,  key=k2)
        self.mu_head = eqx.nn.Linear(d_hidden, d_x, key=k_mu)
        self.log_sigma_head = eqx.nn.Linear(d_hidden, d_x, key=k_sig)

    def _step(self, y_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jax.nn.gelu(self.fc1(y_t))
        x = jax.nn.gelu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = jnp.clip(self.log_sigma_head(x), -8.0, 4.0)
        return mu, log_sigma

    def __call__(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """y: (L, d_y) → mu, log_sigma: (L, d_x)."""
        return jax.vmap(self._step)(y)


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
    Input per step: z_s ‖ mu ‖ c ‖ flow_emb(s) ‖ pos_emb(step/L)
    """
    time_emb:    SinusoidalEmb
    step_emb:    SinusoidalEmb
    input_proj:  eqx.nn.Linear
    layers:      list
    output_proj: eqx.nn.Linear
    condition_on_mu: bool = eqx.field(static=True)

    def __init__(self, d_x: int, d_c: int, d_model: int, n_heads: int,
                 depth: int, time_emb_dim: int, condition_on_mu: bool,
                 key: jr.PRNGKey):
        k_t, k_p, k_in, k_out, *k_layers = jr.split(key, depth + 4)
        self.condition_on_mu = condition_on_mu
        in_size = d_x + d_c + 2 * time_emb_dim + (d_x if condition_on_mu else 0)
        self.time_emb    = SinusoidalEmb(time_emb_dim)
        self.step_emb    = SinusoidalEmb(time_emb_dim)
        self.input_proj  = eqx.nn.Linear(in_size, d_model, key=k_in)
        self.layers      = [TransformerBlock(d_model, n_heads, k_layers[i])
                            for i in range(depth)]
        self.output_proj = eqx.nn.Linear(d_model, d_x, key=k_out)

    def __call__(self, z_s: jnp.ndarray, c: jnp.ndarray, s: jnp.ndarray,
                 mu: jnp.ndarray = None) -> jnp.ndarray:
        """Evaluate the field, optionally conditioned on the inferred mean."""
        L     = z_s.shape[0]
        t_emb = self.time_emb(s)
        t_seq = jnp.broadcast_to(t_emb, (L, t_emb.shape[0]))

        step_norm = jnp.arange(L, dtype=jnp.float32) / jnp.maximum(L - 1, 1)
        p_seq = jax.vmap(self.step_emb)(step_norm)

        parts = [z_s]
        if self.condition_on_mu:
            if mu is None:
                raise ValueError("mu is required when condition_on_mu=True")
            parts.append(mu)
        parts.extend([c, t_seq, p_seq])
        x = jnp.concatenate(parts, axis=-1)
        x = jax.vmap(self.input_proj)(x)
        for layer in self.layers:
            x = layer(x)
        return jax.vmap(self.output_proj)(x)

    def traj(self, z_s, c, s, mu=None):
        return self(z_s, c, s, mu)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StochasticInterpolantSE(eqx.Module):
    """
    Conditional stochastic interpolant for state estimation.

    The source distribution is produced by q_eta(y, c) over the complete state:
    z_0 = mu_eta + sigma_eta * eps_0. No observed/unobserved coordinate masking
    is applied to the source state.
    """
    encoder:            ObsTransformerEncoder
    inference_net:      StateInferenceMLP | None
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
    score_sigma_floor:  float = eqx.field(static=True)
    s_min:              float = eqx.field(static=True)
    interpolant_type:   str   = eqx.field(static=True)
    source_type:        str   = eqx.field(static=True)
    base_mean:          float = eqx.field(static=True)
    base_std:           float = eqx.field(static=True)

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
        score_weight:       float = 0.25,
        lifter_loss_weight: float = 0.1,
        score_sigma_floor:  float = 0.05,
        s_min:              float = 0.02,
        interpolant_type:   str   = "linear",
        source_type:        str   = "inferred",
        base_mean:          float = 0.0,
        base_std:           float = 1.0,
        key:                jr.PRNGKey = jr.PRNGKey(0),
    ):
        if interpolant_type not in ("cosine", "linear"):
            raise ValueError(f"interpolant_type must be 'cosine' or 'linear', got {interpolant_type}")
        if source_type not in ("inferred", "gaussian"):
            raise ValueError(f"source_type must be 'inferred' or 'gaussian', got {source_type}")
        if base_std <= 0:
            raise ValueError("base_std must be positive")
        self.d_x            = d_x
        self.d_y            = d_y
        self.d_c            = d_c
        self.obs_interval   = obs_interval
        self.sigma_0        = sigma_0
        self.g0             = g0
        self.score_weight   = score_weight
        self.lifter_loss_weight = lifter_loss_weight
        self.score_sigma_floor  = score_sigma_floor
        self.s_min          = s_min
        self.interpolant_type = interpolant_type
        self.source_type = source_type
        self.base_mean = base_mean
        self.base_std = base_std

        k_enc, k_inf, k_vel, k_score = jr.split(key, 4)
        self.encoder   = ObsTransformerEncoder(d_y, d_c, d_hidden, n_heads, depth, time_emb_dim, k_enc)
        inferred = source_type == "inferred"
        self.inference_net = StateInferenceMLP(d_y, d_x, d_hidden, k_inf) if inferred else None
        self.vel_net = SITransformerNet(d_x, d_c, d_hidden, n_heads, depth,
                                        time_emb_dim, inferred, k_vel)
        self.score_net = SITransformerNet(d_x, d_c, d_hidden, n_heads, depth,
                                          time_emb_dim, inferred, k_score)

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
        return linear_interpolant_coefficients(s, self.sigma_0)[0]

    def _beta_linear(self, s):
        return linear_interpolant_coefficients(s, self.sigma_0)[1]

    def _alpha_dot_linear(self, s):
        return linear_interpolant_coefficients(s, self.sigma_0)[3]

    def _beta_dot_linear(self, s):
        return linear_interpolant_coefficients(s, self.sigma_0)[4]

    def _sigma_linear(self, s):
        # σ_0 · (1-s) · s = σ_0 · s(1-s) — zero at both endpoints
        return linear_interpolant_coefficients(s, self.sigma_0)[2]

    def _sigma_dot_linear(self, s):
        # d/ds [σ_0 · s(1-s)] = σ_0 · (1 - 2s)
        return linear_interpolant_coefficients(s, self.sigma_0)[5]

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

    def _unscale_score(self, scaled_score: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        """Convert the sigma-scaled score target (-eps) into an actual score."""
        return unscale_interpolant_score(
            scaled_score, self._sigma(s), self.score_sigma_floor
        )

    # ------------------------------------------------------------------
    # Inference-network source: all state coordinates from q_eta
    # ------------------------------------------------------------------

    def _make_source(
        self,
        mu:     jnp.ndarray,   # (L, d_x)
        log_sigma: jnp.ndarray,# (L, d_x)
        key:    jr.PRNGKey,
    ) -> jnp.ndarray:          # (L, d_x)  z_0 with stochastic unobs dims
        noise = jr.normal(key, mu.shape)
        return mu + jnp.exp(log_sigma) * noise

    def _source_nll(self, x: jnp.ndarray, mu: jnp.ndarray, log_sigma: jnp.ndarray) -> jnp.ndarray:
        """Diagonal Gaussian NLL up to an additive constant."""
        inv_sigma = jnp.exp(-log_sigma)
        return jnp.mean(0.5 * ((x - mu) * inv_sigma) ** 2 + log_sigma)

    def _source_params(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return source parameters without leaking encoder context into q_eta."""
        if self.source_type == "inferred":
            return self.inference_net(y)
        shape = (y.shape[0], self.d_x)
        return (
            jnp.full(shape, self.base_mean, dtype=y.dtype),
            jnp.full(shape, jnp.log(self.base_std), dtype=y.dtype),
        )

    def _field(self, net, z, c, s, mu):
        if self.source_type == "inferred":
            return net.traj(z, c, s, mu=mu)
        return net.traj(z, c, s)

    # ------------------------------------------------------------------
    # Core SDE simulation primitive  (one particle, one segment)
    # ------------------------------------------------------------------

    def _simulate_sde_segment(
        self,
        z0:         jnp.ndarray,   # (L, d_x)  initial trajectory (with source noise)
        mu_seg:     jnp.ndarray,   # (L, d_x)  inference mean — conditioning for nets
        c_seg:      jnp.ndarray,   # (L, d_c)  BiGRU context
        key:        jr.PRNGKey,
        n_steps:    int,
        g0:         float,
    ) -> jnp.ndarray:              # (L, d_x)  refined trajectory
        """
        Integrate the Refinement SDE from s=0 to s=1 for one particle.
        mu_seg is passed separately so the networks always see the same
        conditioning signal regardless of the stochastic z_0.
        """
        ds        = 1.0 / n_steps
        s_vals    = jnp.linspace(0.0, 1.0 - ds, n_steps)
        step_keys = jr.split(key, n_steps)

        def em_step(z, inputs):
            s_i, noise_k = inputs
            v         = self._field(self.vel_net, z, c_seg, s_i, mu_seg)
            score     = self._unscale_score(
                self._field(self.score_net, z, c_seg, s_i, mu_seg),
                s_i,
            )
            score_weight, diff_coef = refinement_sde_coefficients(
                self._sigma(s_i), g0 ** 2
            )
            drift     = v + score_weight * score
            noise     = diff_coef * jr.normal(noise_k, z.shape) * jnp.sqrt(ds)
            return z + drift * ds + noise, None

        z1, _ = jax.lax.scan(em_step, z0, (s_vals, step_keys))
        return z1

    # ------------------------------------------------------------------
    # Core loss primitive  (one segment, pre-computed context + q_eta params)
    # ------------------------------------------------------------------

    def _loss_seg_components(
        self,
        x_seg:      jnp.ndarray,   # (L, d_x)  ground-truth state segment
        mu_seg:     jnp.ndarray,   # (L, d_x)  inference mean
        log_sigma_seg: jnp.ndarray,# (L, d_x)  inference log std
        c_seg:      jnp.ndarray,   # (L, d_c)  encoded context segment
        key:        jr.PRNGKey,
    ) -> jnp.ndarray:
        """
        Joint flow + σ-scaled score loss.

        Source z_0: all state coordinates sampled from q_eta(y, c).
        Velocity target: (x - z_0) + σ_dot * ε  — depends on z_0, hence on key.
        Score target: -ε  — independent of z_0.
        """
        k_s, k_eps, k_z0 = jr.split(key, 3)
        s   = jr.uniform(k_s, minval=self.s_min, maxval=1.0 - self.s_min)
        eps = jr.normal(k_eps, x_seg.shape)

        z0_seg = self._make_source(mu_seg, log_sigma_seg, k_z0)   # (L, d_x)

        sig_s  = self._sigma(s)
        z_s    = self._alpha(s) * z0_seg + self._beta(s) * x_seg + sig_s * eps

        v_target   = self._alpha_dot(s) * z0_seg + self._beta_dot(s) * x_seg + self._sigma_dot(s) * eps
        v_pred     = self._field(self.vel_net, z_s, c_seg, s, mu_seg)
        l_flow     = jnp.mean((v_pred - v_target) ** 2)

        score_pred = self._field(self.score_net, z_s, c_seg, s, mu_seg)
        l_score    = jnp.mean((score_pred + eps) ** 2)

        l_source = (self._source_nll(x_seg, mu_seg, log_sigma_seg)
                    if self.source_type == "inferred" else jnp.array(0.0))

        total = l_flow + self.score_weight * l_score + self.lifter_loss_weight * l_source
        return total, l_flow, l_score, l_source

    def _loss_seg(
        self,
        x_seg:      jnp.ndarray,
        mu_seg:     jnp.ndarray,
        log_sigma_seg: jnp.ndarray,
        c_seg:      jnp.ndarray,
        key:        jr.PRNGKey,
    ) -> jnp.ndarray:
        return self._loss_seg_components(x_seg, mu_seg, log_sigma_seg, c_seg, key)[0]

    # ------------------------------------------------------------------
    # Full-trajectory training loss
    # ------------------------------------------------------------------

    def loss_single(self, x, y, key):
        """Full-trajectory loss for one (x, y) pair."""
        c = self.encoder(y)
        mu, log_sigma = self._source_params(y)
        return self._loss_seg(x, mu, log_sigma, c, key)

    def loss_single_components(self, x, y, key):
        """Full-trajectory loss components for one (x, y) pair."""
        c = self.encoder(y)
        mu, log_sigma = self._source_params(y)
        return self._loss_seg_components(x, mu, log_sigma, c, key)

    def loss(self, x, y, key):
        """Batch full-trajectory loss. x: (B,T,d_x)  y: (B,T,d_y)."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(self.loss_single)(x, y, keys))

    def loss_components(self, x, y, key):
        """Batch full-trajectory loss plus diagnostic components."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        vals = jax.vmap(self.loss_single_components)(x, y, keys)
        return tuple(jnp.mean(v) for v in vals)

    # ------------------------------------------------------------------
    # Multiple-shooting training loss
    # ------------------------------------------------------------------

    def loss_ms_single(
        self,
        x:       jnp.ndarray,   # (T, d_x)
        y:       jnp.ndarray,   # (T, d_y)
        key:     jr.PRNGKey,
        seg_len: int,
        segment_stride: int = 0,
    ) -> jnp.ndarray:
        T      = x.shape[0]
        stride = segment_stride if segment_stride > 0 else seg_len
        n_segs = (T - seg_len - 1) // stride + 1
        L      = seg_len + 1

        c_full = self.encoder(y)
        mu_full, log_sigma_full = self._source_params(y)

        seg_starts = (jnp.arange(n_segs) * stride).astype(jnp.int32)

        x_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(x,          (s, 0), (L, self.d_x)))(seg_starts)
        mu_segs = jax.vmap(lambda s: jax.lax.dynamic_slice(mu_full, (s, 0), (L, self.d_x)))(seg_starts)
        log_sigma_segs = jax.vmap(lambda s: jax.lax.dynamic_slice(log_sigma_full, (s, 0), (L, self.d_x)))(seg_starts)
        c_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(c_full,      (s, 0), (L, self.d_c)))(seg_starts)

        keys   = jr.split(key, n_segs)
        losses = jax.vmap(self._loss_seg)(x_segs, mu_segs, log_sigma_segs, c_segs, keys)
        return jnp.mean(losses)

    def loss_ms_single_components(
        self,
        x:       jnp.ndarray,
        y:       jnp.ndarray,
        key:     jr.PRNGKey,
        seg_len: int,
        segment_stride: int = 0,
    ):
        T      = x.shape[0]
        stride = segment_stride if segment_stride > 0 else seg_len
        n_segs = (T - seg_len - 1) // stride + 1
        L      = seg_len + 1

        c_full = self.encoder(y)
        mu_full, log_sigma_full = self._source_params(y)

        seg_starts = (jnp.arange(n_segs) * stride).astype(jnp.int32)

        x_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(x,          (s, 0), (L, self.d_x)))(seg_starts)
        mu_segs = jax.vmap(lambda s: jax.lax.dynamic_slice(mu_full, (s, 0), (L, self.d_x)))(seg_starts)
        log_sigma_segs = jax.vmap(lambda s: jax.lax.dynamic_slice(log_sigma_full, (s, 0), (L, self.d_x)))(seg_starts)
        c_segs      = jax.vmap(lambda s: jax.lax.dynamic_slice(c_full,      (s, 0), (L, self.d_c)))(seg_starts)

        keys = jr.split(key, n_segs)
        vals = jax.vmap(self._loss_seg_components)(x_segs, mu_segs, log_sigma_segs, c_segs, keys)
        return tuple(jnp.mean(v) for v in vals)

    def loss_ms_batch(self, x, y, key, seg_len, segment_stride=0):
        """Batch multiple-shooting loss."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.loss_ms_single(
                xi, yi, ki, seg_len, segment_stride
            )
        )(x, y, keys))

    def loss_ms_batch_components(self, x, y, key, seg_len, segment_stride=0):
        """Batch multiple-shooting loss plus diagnostic components."""
        B    = x.shape[0]
        keys = jr.split(key, B)
        vals = jax.vmap(
            lambda xi, yi, ki: self.loss_ms_single_components(
                xi, yi, ki, seg_len, segment_stride
            )
        )(x, y, keys)
        return tuple(jnp.mean(v) for v in vals)

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
        Each particle draws its own z_0 ~ q_eta(y, c) → diverse posterior.
        Returns (n_particles, T, d_x).
        """
        c = self.encoder(y)
        mu, log_sigma = self._source_params(y)
        g0 = self.g0 if g0_override is None else g0_override

        def simulate_one(pk):
            k_z0, k_sde = jr.split(pk)
            z0 = self._make_source(mu, log_sigma, k_z0)
            return self._simulate_sde_segment(z0, mu, c, k_sde, n_steps, g0)

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
        Each particle draws its own z_0 from q_eta for diversity.
        Returns (n_particles, T, d_x).
        """
        ds     = 1.0 / n_steps
        c = self.encoder(y)
        mu, log_sigma = self._source_params(y)
        s_vals = jnp.linspace(0.0, 1.0 - ds, n_steps)

        def simulate_one(pk):
            z0 = self._make_source(mu, log_sigma, pk)
            def euler_step(z, s_i):
                v = self._field(self.vel_net, z, c, s_i, mu)
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

        At each segment k, each particle draws its own z_0 ~ q_eta(y, c)
        for the interior time steps, giving diverse posterior trajectories.
        The boundary (t=0 of each segment) carries the last refined state
        from segment k-1.

        Returns (n_particles, n_segs*seg_len+1, d_x).
        """
        T      = y.shape[0]
        n_segs = (T - 1) // seg_len
        L      = seg_len + 1
        g0     = self.g0 if g0_override is None else g0_override

        c_full = self.encoder(y)
        mu_full, log_sigma_full = self._source_params(y)

        def process_segment(boundary, k):
            """
            boundary : (n_particles, d_x) — last refined state of prev segment
            k        : scalar segment index
            """
            start      = k * seg_len
            c_seg      = jax.lax.dynamic_slice(c_full,      (start, 0), (L, self.d_c))
            mu_seg     = jax.lax.dynamic_slice(mu_full, (start, 0), (L, self.d_x))
            log_sigma_seg = jax.lax.dynamic_slice(log_sigma_full, (start, 0), (L, self.d_x))

            p_keys = jr.split(jr.fold_in(key, k), n_particles)

            def sim_one(bnd, pk):
                k_z0, k_sde = jr.split(pk)
                # Interior steps t=1..L-1: stochastic source from q_eta
                z0_rest = self._make_source(mu_seg[1:], log_sigma_seg[1:], k_z0)
                # First step t=0: boundary from previous segment (carries refined state)
                z0_all      = jnp.concatenate([bnd[None], z0_rest], axis=0)  # (L, d_x)
                return self._simulate_sde_segment(z0_all, mu_seg, c_seg, k_sde, n_steps, g0)

            seg_out      = jax.vmap(sim_one)(boundary, p_keys)   # (n_p, L, d_x)
            new_boundary = seg_out[:, -1, :]                      # (n_p, d_x)
            return new_boundary, seg_out

        # Segment 0: first time-step boundary is sampled from q_eta.
        k_bnd0, key = jr.split(key)
        boundary_0 = jax.vmap(
            lambda k: self._make_source(mu_full[:1], log_sigma_full[:1], k)[0]
        )(jr.split(k_bnd0, n_particles))

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

    def sample_posterior_segmented_parallel(
        self,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        n_particles: int   = 100,
        n_steps:     int   = 100,
        seg_len:     int   = 20,
        g0_override: float = None,
    ) -> jnp.ndarray:
        """Infer all trajectory segments and particles in parallel.

        Unlike :meth:`sample_posterior_segmented`, this sampler does not carry
        a refined boundary from segment ``k`` into segment ``k+1``. Each
        segment draws its complete Gaussian source independently, which makes
        the segment axis eligible for ``vmap`` and removes the outer sequential
        scan. Overlapping boundary states are stitched by retaining the first
        segment's complete output and dropping the first state of later ones.
        """
        T = y.shape[0]
        n_segs = (T - 1) // seg_len
        L = seg_len + 1
        g0 = self.g0 if g0_override is None else g0_override

        c_full = self.encoder(y)
        mu_full, log_sigma_full = self._source_params(y)
        starts = (jnp.arange(n_segs) * seg_len).astype(jnp.int32)
        c_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(c_full, (s, 0), (L, self.d_c))
        )(starts)
        mu_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(mu_full, (s, 0), (L, self.d_x))
        )(starts)
        log_sigma_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(
                log_sigma_full, (s, 0), (L, self.d_x)
            )
        )(starts)
        segment_keys = jr.split(key, n_segs)

        def sample_segment(mu_seg, log_sigma_seg, c_seg, segment_key):
            def sample_particle(particle_key):
                source_key, sde_key = jr.split(particle_key)
                z0 = self._make_source(mu_seg, log_sigma_seg, source_key)
                return self._simulate_sde_segment(
                    z0, mu_seg, c_seg, sde_key, n_steps, g0
                )

            return jax.vmap(sample_particle)(
                jr.split(segment_key, n_particles)
            )

        segments = jax.vmap(sample_segment)(
            mu_segs, log_sigma_segs, c_segs, segment_keys
        )  # (n_segs, n_particles, L, d_x)
        first = segments[0]
        rest = jnp.swapaxes(segments[1:, :, 1:, :], 0, 1)
        rest = rest.reshape(
            n_particles, (n_segs - 1) * seg_len, self.d_x
        )
        return jnp.concatenate([first, rest], axis=1)

    def sample_posterior_ode_segmented_parallel(
        self,
        y:           jnp.ndarray,
        key:         jr.PRNGKey,
        n_particles: int = 100,
        n_steps:     int = 100,
        seg_len:     int = 20,
    ) -> jnp.ndarray:
        """Velocity-only ODE sampling with independent parallel segments."""
        T = y.shape[0]
        n_segs = (T - 1) // seg_len
        L = seg_len + 1
        c_full = self.encoder(y)
        mu_full, log_sigma_full = self._source_params(y)
        starts = (jnp.arange(n_segs) * seg_len).astype(jnp.int32)
        c_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(c_full, (s, 0), (L, self.d_c))
        )(starts)
        mu_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(mu_full, (s, 0), (L, self.d_x))
        )(starts)
        log_sigma_segs = jax.vmap(
            lambda s: jax.lax.dynamic_slice(
                log_sigma_full, (s, 0), (L, self.d_x)
            )
        )(starts)
        segment_keys = jr.split(key, n_segs)
        ds = 1.0 / n_steps
        s_values = jnp.linspace(0.0, 1.0 - ds, n_steps)

        def sample_segment(mu_seg, log_sigma_seg, c_seg, segment_key):
            def sample_particle(particle_key):
                z0 = self._make_source(mu_seg, log_sigma_seg, particle_key)

                def step(z, s):
                    return z + ds * self._field(
                        self.vel_net, z, c_seg, s, mu_seg
                    ), None

                final, _ = jax.lax.scan(step, z0, s_values)
                return final

            return jax.vmap(sample_particle)(
                jr.split(segment_key, n_particles)
            )

        segments = jax.vmap(sample_segment)(
            mu_segs, log_sigma_segs, c_segs, segment_keys
        )
        first = segments[0]
        rest = jnp.swapaxes(segments[1:, :, 1:, :], 0, 1)
        rest = rest.reshape(
            n_particles, (n_segs - 1) * seg_len, self.d_x
        )
        return jnp.concatenate([first, rest], axis=1)

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
    def loss_fn(m):
        total, l_flow, l_score, l_source = m.loss_components(x_batch, y_batch, key)
        return total, (l_flow, l_score, l_source)
    (loss, components), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss, components


@eqx.filter_jit
def make_step_si_ms(
    model:     StochasticInterpolantSE,
    x_batch:   jnp.ndarray,
    y_batch:   jnp.ndarray,
    key:       jr.PRNGKey,
    seg_len:   int,
    segment_stride: int,
    opt_state,
    optimizer,
):
    """Multiple-shooting training step."""
    def loss_fn(m):
        total, l_flow, l_score, l_source = m.loss_ms_batch_components(
            x_batch, y_batch, key, seg_len, segment_stride
        )
        return total, (l_flow, l_score, l_source)
    (loss, components), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss, components
