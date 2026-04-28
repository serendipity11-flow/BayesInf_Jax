"""
LatentSSMFlow: Conditional Normalizing Flow with Linear Gaussian SSM latent dynamics
for full-sequence state estimation.

Architecture (state estimation):

    enc(y_{1:T}) ── cond_t ──┐
    x_{t-1}  ────────────────┤
                              │
    z_1 ~ N(0, P_inf)         │
    z_t = F z_{t-1} + G ε_t  flow⁻¹(·; cond_t, x_{t-1}) ──► x_t

Segmentation (seg_len > 0):
    - Trajectory split into n_segs = T // seg_len segments.
    - Within each segment: autoregressive x_{t-1} conditioning (scan over seg_len steps).
    - Across segments: fully parallel (vmap over n_segs).
    - SSM prior restarts independently at each segment boundary (z_1 ~ N(0, P_inf)).
    - Segment boundary x_prev:
        * Training  — true state at the last step of the preceding segment.
        * Inference — true state at the last step of the preceding segment (seg_starts).

Training — exact MLE via change of variables:
    x_prev_t       = true state at t-1 within segment
    z_t            = flow(x_t ; [cond_t, x_prev_t])
    log p(x_{1:T}) = Σ_segs [ Σ_t log|det J| + log p_SSM(z_{1:seg}) ]

Inference (posterior samples):
    z_{1:seg} ~ SSM prior  (per segment, independent)
    x_t = flow⁻¹(z_t; [cond_t, x_{t-1}])  (scan within segment, vmap across segments)
    Boundary x_prev for segment s provided via seg_starts (true state x_{s*seg_len - 1}).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Tuple

from bayesinf.applications.state_estimation.amortized_state_flow import (
    StateConditionEncoder,
)
from bayesinf.applications.state_estimation.pointwise_state_flow import (
    build_flow_transforms,
)
from bayesinf.models.normalizing_flow import ActNorm, RandomPermutation, MAFLayer


# ---------------------------------------------------------------------------
# Flow forward / inverse helpers  (single timestep)
# ---------------------------------------------------------------------------

def _flow_fwd_single(
    transforms: list,
    x:          jnp.ndarray,   # (d_x,)
    x_prev:     jnp.ndarray,   # (d_x,)
    cond:       jnp.ndarray,   # (d_out,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """x → z  (encoding). Returns (z, log|dz/dx|)."""
    combined_cond = jnp.concatenate([x_prev, cond])
    z_b  = x[None]
    ld_b = jnp.zeros((1,))
    for t in transforms:
        if isinstance(t, (ActNorm, RandomPermutation)):
            z_b, ldj = t(z_b, reverse=False)
        else:
            z_b, ldj = t(z_b, cond=combined_cond[None], reverse=False)
        ld_b = ld_b + ldj
    return z_b[0], ld_b[0]


def _flow_inv_single(
    transforms: list,
    z:          jnp.ndarray,   # (d_x,)
    x_prev:     jnp.ndarray,   # (d_x,)
    cond:       jnp.ndarray,   # (d_out,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """z → x  (decoding). Returns (x, log|dx/dz|)."""
    combined_cond = jnp.concatenate([x_prev, cond])
    x_b  = z[None]
    ld_b = jnp.zeros((1,))
    for t in reversed(transforms):
        if isinstance(t, (ActNorm, RandomPermutation)):
            x_b, ldj = t(x_b, reverse=True)
        else:
            x_b, ldj = t(x_b, cond=combined_cond[None], reverse=True)
        ld_b = ld_b + ldj
    return x_b[0], ld_b[0]


# ---------------------------------------------------------------------------
# LatentSSMFlow
# ---------------------------------------------------------------------------

class LatentSSMFlow(eqx.Module):
    """
    Linear Gaussian SSM in latent space + conditional normalizing flow decoder.

    Parameters
    ----------
    d_x          : state / data dimension
    d_y          : observation dimension (defaults to d_x if -1)
    d_u          : control input dimension (0 if unused)
    d_enc        : encoder internal hidden dim
    d_out        : flow conditioning dim (encoder output)
    seg_len      : segment length for parallel processing (0 = full sequence)
    encoder_type : 'gru' | 'transformer' | 'pointwise'
                   'pointwise' — each step t is conditioned only on (y_t, u_t)
                   via a small MLP; no future information leaks into cond_t.
                   Temporal context comes entirely from the SSM and x_{t-1}.
    flow_type    : 'realnvp' | 'nsf' | 'maf'
    n_coupling   : number of flow coupling blocks
    hidden_size  : MLP hidden width (flow)
    d_enc        : hidden width of the pointwise encoder MLP
    n_hidden     : MLP depth
    max_rho      : spectral radius cap on F
    """

    encoder:    object   # StateConditionEncoder  OR  eqx.nn.MLP (pointwise)
    transforms: list

    F:      jnp.ndarray   # (d_x, d_x)  transition matrix (raw; use F_stable)
    G_chol: jnp.ndarray   # (d_x, d_x)  lower-tri Cholesky of Q = G Gᵀ

    d_x:        int   = eqx.field(static=True)
    d_y:        int   = eqx.field(static=True)
    d_u:        int   = eqx.field(static=True)
    d_out:      int   = eqx.field(static=True)
    seg_len:    int   = eqx.field(static=True)
    max_rho:    float = eqx.field(static=True)
    pointwise:  bool  = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_y:          int        = -1,
        d_u:          int        = 0,
        d_enc:        int        = 64,
        d_out:        int        = 96,
        seg_len:      int        = 0,   # 0 = full sequence (no segmentation)
        encoder_type: str        = "transformer",
        local_window: bool       = False,
        gru_hidden:   int        = 64,
        d_model:      int        = 64,
        n_heads:      int        = 4,
        d_ff:         int        = 256,
        n_layers:     int        = 3,
        flow_type:    str        = "realnvp",
        n_coupling:   int        = 8,
        hidden_size:  int        = 128,
        n_hidden:     int        = 2,
        num_bins:     int        = 8,
        max_rho:      float      = 0.95,
        key:          jr.PRNGKey = jr.PRNGKey(0),
    ):
        self.d_x      = d_x
        self.d_y      = d_x if d_y < 0 else d_y
        self.d_u      = d_u
        self.d_out    = d_out
        self.seg_len  = seg_len
        self.max_rho  = max_rho
        self.pointwise = (encoder_type == "pointwise")

        k_enc, k_flow, k_F, k_G = jr.split(key, 4)

        if encoder_type == "pointwise":
            # ── Pointwise encoder: cond_t = MLP(y_t, u_t) ────────────────
            # Each step is conditioned ONLY on the local observation y_t and
            # control u_t.  No future information enters cond_t; temporal
            # context is carried entirely by the SSM latent and x_{t-1}.
            self.encoder = eqx.nn.MLP(
                in_size    = self.d_y + max(d_u, 1),
                out_size   = d_out,
                width_size = d_enc,
                depth      = n_hidden,
                key        = k_enc,
            )
        else:
            # ── Full-sequence encoder (transformer / GRU) ─────────────────
            self.encoder = StateConditionEncoder(
                d_y          = self.d_y,
                d_u          = d_u,
                d_x          = d_x,
                d_enc        = d_enc,
                d_out        = d_out,
                key          = k_enc,
                encoder_type = encoder_type,
                local_window = local_window,
                gru_hidden   = gru_hidden,
                d_model      = d_model,
                n_heads      = n_heads,
                d_ff         = d_ff,
                n_layers     = n_layers,
            )

        # ── Conditional flow: z_t ↔ x_t | [x_{t-1}, cond_t] ─────────────
        self.transforms = build_flow_transforms(
            input_dim   = d_x,
            cond_dim    = d_out + d_x,   # encoder output + previous state
            n_blocks    = n_coupling,
            hidden_size = hidden_size,
            n_hidden    = n_hidden,
            flow_type   = flow_type,
            key         = k_flow,
            num_bins    = num_bins,
        )

        # ── Linear Gaussian SSM ───────────────────────────────────────────
        # F init: ρ=0.8 diagonal — faster decay suited to chaotic systems
        # (0.8^seg_len ≈ 0.33 residual memory per segment, vs 0.77 at ρ=0.95).
        self.F      = 0.80 * jnp.eye(d_x) + 0.05 * jr.normal(k_F, (d_x, d_x))
        # G_chol init: larger noise (1.0*I) to give a wider SSM prior,
        # which helps avoid posterior undercoverage.
        raw_G       = jr.normal(k_G, (d_x, d_x)) * 0.1
        self.G_chol = jnp.tril(raw_G) + 1.0 * jnp.eye(d_x)

    # ------------------------------------------------------------------
    # Encoding helper  (dispatches pointwise vs full-sequence encoder)
    # ------------------------------------------------------------------

    def _encode(
        self,
        y:  jnp.ndarray,   # (T, d_y)
        u:  jnp.ndarray,   # (T, d_u)
        x0: jnp.ndarray,   # (d_x,)
    ) -> jnp.ndarray:      # (T, d_out)
        """
        Returns per-step conditioning vectors cond_{1:T}.

        pointwise  — cond_t = MLP([y_t, u_t]).  Each step sees only its own
                     observation; no future information leaks in.
        sequence   — cond_t = Encoder(y_{1:T}, u_{1:T}, x0)_t.  Full smoother.
        """
        if self.pointwise:
            u_in = u if self.d_u > 0 else jnp.zeros((y.shape[0], 1))
            inp  = jnp.concatenate([y, u_in], axis=-1)   # (T, d_y + d_u)
            return jax.vmap(self.encoder)(inp)            # (T, d_out)
        else:
            return self.encoder(y[None], u[None], x0[None])[0]   # (T, d_out)

    # ------------------------------------------------------------------
    # SSM parameter helpers
    # ------------------------------------------------------------------

    @property
    def F_stable(self) -> jnp.ndarray:
        """F rescaled so spectral radius ≤ max_rho."""
        rho   = jnp.max(jnp.abs(jnp.linalg.eigvals(self.F)))
        scale = jnp.minimum(self.max_rho / jax.lax.stop_gradient(rho), 1.0)
        return self.F * scale

    @staticmethod
    def _to_chol(raw: jnp.ndarray) -> jnp.ndarray:
        L = jnp.tril(raw, k=-1)
        d = jax.nn.softplus(jnp.diag(raw)) + 1e-4
        return L + jnp.diag(d)

    @property
    def G_chol_pd(self) -> jnp.ndarray:
        return self._to_chol(self.G_chol)

    @property
    def Q(self) -> jnp.ndarray:
        L = self.G_chol_pd
        return L @ L.T

    @property
    def P_inf(self) -> jnp.ndarray:
        """Stationary covariance via Kronecker solve: (I - F⊗F) vec(P) = vec(Q)."""
        F, Q = self.F_stable, self.Q
        n = self.d_x
        A = jnp.eye(n * n) - jnp.kron(F, F)
        return jnp.linalg.solve(A, Q.ravel()).reshape(n, n)

    # ------------------------------------------------------------------
    # SSM log-probability  log p(z_{1:T})
    # ------------------------------------------------------------------

    def _ssm_log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        log N(z_1; 0, P_inf) + Σ_{t=2}^{T} log N(z_t; F z_{t-1}, Q).
        z : (T, d_x)
        """
        d   = self.d_x
        G_L = self.G_chol_pd
        F   = self.F_stable

        # z_1 ~ N(0, P_inf)  — stationary distribution
        L_inf   = jnp.linalg.cholesky(self.P_inf + 1e-6 * jnp.eye(d))
        z1      = z[0]
        a1      = jax.scipy.linalg.solve_triangular(L_inf, z1, lower=True)
        lp_z1   = -0.5 * (jnp.dot(a1, a1) + d * jnp.log(2.0 * jnp.pi)
                          + 2.0 * jnp.sum(jnp.log(jnp.diag(L_inf))))

        z_prev = z[:-1]
        z_curr = z[1:]
        z_pred = jax.vmap(lambda zp: F @ zp)(z_prev)
        resid  = z_curr - z_pred
        alpha  = jax.vmap(
            lambda r: jax.scipy.linalg.solve_triangular(G_L, r, lower=True)
        )(resid)
        log_det  = jnp.sum(jnp.log(jnp.diag(G_L)))
        lp_trans = -0.5 * (
            jnp.sum(alpha ** 2, axis=-1)
            + d * jnp.log(2.0 * jnp.pi)
            + 2.0 * log_det
        )
        return lp_z1 + jnp.sum(lp_trans)

    # ------------------------------------------------------------------
    # Internal: encode one segment  (parallel vmap over seg_len steps)
    # ------------------------------------------------------------------

    def _encode_segment(
        self,
        x_seg:    jnp.ndarray,   # (seg, d_x)  true states in segment
        cond_seg: jnp.ndarray,   # (seg, d_out)
        x_start:  jnp.ndarray,   # (d_x,)  boundary state before segment
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward flow over one segment — multiple-shooting autoregressive scan.

        The first step uses x_start (true boundary) as x_prev.
        Every subsequent step uses the DECODED output of the previous step as
        x_prev instead of the true previous state.  This mirrors the
        autoregressive decoding used at inference time (multiple-shooting
        training: true anchor at each patch boundary, free-run within patch).

        Returns (z_seg, logdet_seg), both (seg,*).
        """
        def encode_step(x_prev, inputs):
            x_t, cond_t = inputs
            z_t, ld_t   = _flow_fwd_single(self.transforms, x_t, x_prev, cond_t)
            # Decode z_t back → x̂_t becomes x_prev for the next step
            x_hat_t, _  = _flow_inv_single(self.transforms, z_t, x_prev, cond_t)
            return x_hat_t, (z_t, ld_t)

        _, (z, logdet) = jax.lax.scan(encode_step, x_start, (x_seg, cond_seg))
        return z, logdet

    # ------------------------------------------------------------------
    # Single-sequence log p(x_{1:T})
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x:  jnp.ndarray,   # (T, d_x)
        y:  jnp.ndarray,   # (T, d_y)
        u:  jnp.ndarray,   # (T, d_u)
        x0: jnp.ndarray,   # (d_x,)
    ) -> jnp.ndarray:
        T    = x.shape[0]
        cond = self._encode(y, u, x0)   # (T, d_out)
        seg  = self.seg_len

        if seg <= 0 or T <= seg:
            # ── Full-sequence: one segment ────────────────────────────────
            z, logdet = self._encode_segment(x, cond, x0)
            return jnp.sum(logdet) + self._ssm_log_prob(z)

        # ── Segmented: parallel encoding, independent SSM per segment ────
        n_segs = T // seg

        # Boundary state before segment s: s=0 → x0, s>0 → true last state
        seg_starts = jnp.concatenate(
            [x0[None], x[seg - 1 : n_segs * seg - 1 : seg]], axis=0
        )  # (n_segs, d_x)

        x_segs    = x[:n_segs * seg].reshape(n_segs, seg, self.d_x)
        cond_segs = cond[:n_segs * seg].reshape(n_segs, seg, self.d_out)

        z_segs, ld_segs = jax.vmap(self._encode_segment)(x_segs, cond_segs, seg_starts)
        # Independent SSM log-prob per segment (each starts fresh from N(0, P_inf))
        lp_ssm = jnp.sum(jax.vmap(self._ssm_log_prob)(z_segs))
        return jnp.sum(ld_segs) + lp_ssm

    # ------------------------------------------------------------------
    # Batch NLL loss
    # ------------------------------------------------------------------

    def nll_loss(
        self,
        x:  jnp.ndarray,   # (M, T, d_x)
        y:  jnp.ndarray,   # (M, T, d_y)
        u:  jnp.ndarray,   # (M, T, d_u)
        x0: jnp.ndarray,   # (M, d_x)
    ) -> jnp.ndarray:
        """Batch NLL: -mean_M [ (1/T) log p(x_{1:T}) ]."""
        T         = x.shape[1]
        log_probs = jax.vmap(self.log_prob)(x, y, u, x0)
        return -jnp.mean(log_probs) / T

    # ------------------------------------------------------------------
    # Entropy regularizer
    # ------------------------------------------------------------------

    def _entropy_reg_single(
        self,
        y:         jnp.ndarray,   # (T, d_y)
        u:         jnp.ndarray,   # (T, d_u)
        x0:        jnp.ndarray,   # (d_x,)
        key:       jr.PRNGKey,
        n_samples: int = 4,
    ) -> jnp.ndarray:
        """
        Estimate  -H[q(x_{1:T} | y_{1:T})]  for one sequence via Monte Carlo.

        Decomposition:
            H[q] = H[p_SSM(z)] + E_{z ~ p_SSM}[ Σ_t log|J⁻¹_t(z_t; cond_t, x̂_{t-1})| ]

        The second term is estimated by sampling z from the SSM prior, running
        the autoregressive inverse flow, and summing the log|J⁻¹| at each step.
        The first term (H[p_SSM]) depends only on the SSM parameters and provides
        its own gradient signal; we include it analytically below.

        Returns a scalar to be added to NLL with weight +lambda_ent.
        Minimising it (= maximising H[q]) widens the posterior.
        """
        T   = y.shape[0]
        d   = self.d_x
        cond = self._encode(y, u, x0)   # (T, d_out)
        G_L  = self.G_chol_pd
        F    = self.F_stable
        L_inf = jnp.linalg.cholesky(self.P_inf + 1e-6 * jnp.eye(d))

        # ── SSM entropy (analytical) ───────────────────────────────────────
        # H[p_SSM] = (1/2) log det P_inf + (T-1)/2 log det Q + (T*d/2)(1 + log 2π)
        log_det_P_inf = 2.0 * jnp.sum(jnp.log(jnp.diag(L_inf)))
        log_det_Q     = 2.0 * jnp.sum(jnp.log(jnp.diag(G_L)))
        h_ssm = (
            0.5 * log_det_P_inf
            + 0.5 * (T - 1) * log_det_Q
            + 0.5 * T * d * (1.0 + jnp.log(2.0 * jnp.pi))
        )

        # ── E_{z~p_SSM}[ Σ_t log|J⁻¹_t| ] via MC ─────────────────────────
        def one_sample(k):
            keys_z = jr.split(k, T)
            z1     = L_inf @ jr.normal(keys_z[0], (d,))

            def ssm_step(z_p, k_i):
                z_t = F @ z_p + G_L @ jr.normal(k_i, (d,))
                return z_t, z_t

            _, z_rest = jax.lax.scan(ssm_step, z1, keys_z[1:])
            z_all     = jnp.concatenate([z1[None], z_rest], axis=0)   # (T, d)

            def decode_step(x_p, inp):
                z_t, c_t = inp
                x_t, ldj = _flow_inv_single(self.transforms, z_t, x_p, c_t)
                return x_t, ldj   # ldj = log|J⁻¹_t| scalar

            _, ldj_all = jax.lax.scan(decode_step, x0, (z_all, cond))
            return jnp.sum(ldj_all)   # Σ_t log|J⁻¹_t|

        mean_ldj = jnp.mean(jax.vmap(one_sample)(jr.split(key, n_samples)))

        # H[q] = h_ssm + E[Σ log|J⁻¹|];  return -H[q] so minimising = widening q
        return -(h_ssm + mean_ldj)

    def entropy_reg_loss(
        self,
        y:         jnp.ndarray,   # (M, T, d_y)
        u:         jnp.ndarray,   # (M, T, d_u)
        x0:        jnp.ndarray,   # (M, d_x)
        key:       jr.PRNGKey,
        n_samples: int = 4,
    ) -> jnp.ndarray:
        """
        Batch entropy regularizer: mean_M [ -H[q(x|y_m)] ] / T.

        Add to nll_loss with weight lambda_ent:
            loss = model.nll_loss(x, y, u, x0) \
                 + lambda_ent * model.entropy_reg_loss(y, u, x0, key)
        """
        T    = y.shape[1]
        keys = jr.split(key, y.shape[0])
        neg_H = jax.vmap(
            lambda yi, ui, x0i, ki: self._entropy_reg_single(yi, ui, x0i, ki, n_samples)
        )(y, u, x0, keys)
        return jnp.mean(neg_H) / T

    # ------------------------------------------------------------------
    # Encode latents  z_t = flow(x_t; [cond_t, x_{t-1}])
    # ------------------------------------------------------------------

    def encode_latents(
        self,
        x:  jnp.ndarray,
        y:  jnp.ndarray,
        u:  jnp.ndarray,
        x0: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (z, cond), both (T, *)."""
        T    = x.shape[0]
        cond = self._encode(y, u, x0)
        seg  = self.seg_len

        if seg <= 0 or T <= seg:
            z, _ = self._encode_segment(x, cond, x0)
            return z, cond

        n_segs    = T // seg
        seg_starts = jnp.concatenate(
            [x0[None], x[seg - 1 : n_segs * seg - 1 : seg]], axis=0
        )
        x_segs    = x[:n_segs * seg].reshape(n_segs, seg, self.d_x)
        cond_segs = cond[:n_segs * seg].reshape(n_segs, seg, self.d_out)
        z_segs, _ = jax.vmap(self._encode_segment)(x_segs, cond_segs, seg_starts)
        return z_segs.reshape(n_segs * seg, self.d_x), cond

    # ------------------------------------------------------------------
    # Reconstruction  (encode then decode with true x_prev — ~identity)
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        x:  jnp.ndarray,
        y:  jnp.ndarray,
        u:  jnp.ndarray,
        x0: jnp.ndarray,
    ) -> jnp.ndarray:
        """x_t → z_t → x̂_t using true x_prev.  Should be ~identity."""
        T    = x.shape[0]
        cond = self._encode(y, u, x0)
        seg  = self.seg_len

        def _recon_seg(x_s, cond_s, x_start):
            x_prev_s = jnp.concatenate([x_start[None], x_s[:-1]], axis=0)
            z_s, _ = jax.vmap(
                lambda xi, xp, ci: _flow_fwd_single(self.transforms, xi, xp, ci)
            )(x_s, x_prev_s, cond_s)
            x_hat_s, _ = jax.vmap(
                lambda zi, xp, ci: _flow_inv_single(self.transforms, zi, xp, ci)
            )(z_s, x_prev_s, cond_s)
            return x_hat_s

        if seg <= 0 or T <= seg:
            return _recon_seg(x, cond, x0)

        n_segs    = T // seg
        seg_starts = jnp.concatenate(
            [x0[None], x[seg - 1 : n_segs * seg - 1 : seg]], axis=0
        )
        x_segs    = x[:n_segs * seg].reshape(n_segs, seg, self.d_x)
        cond_segs = cond[:n_segs * seg].reshape(n_segs, seg, self.d_out)
        x_hat_segs = jax.vmap(_recon_seg)(x_segs, cond_segs, seg_starts)
        return x_hat_segs.reshape(n_segs * seg, self.d_x)

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        y:          jnp.ndarray,
        u:          jnp.ndarray,
        x0:         jnp.ndarray,
        key:        jr.PRNGKey,
        n_samples:  int           = 1,
        seg_starts: jnp.ndarray   = None,   # (n_segs, d_x)  boundary states
        x:          jnp.ndarray   = None,   # (T, d_x)  true states for MS eval
    ) -> jnp.ndarray:
        """
        Draw posterior samples x_{1:T} | y_{1:T}.  Returns (n_samples, T, d_x).

        Segmented, multiple-shooting evaluation (seg_len > 0 and x provided):
          - Sequence split into n_segs patches, each of seg_len steps.
          - Patches run fully in PARALLEL (vmap over n_segs).
          - Within each patch:
              Step 0  : true first state x[s*seg_len] is ENCODED → z_0
                        (anchors the patch to the true state, no SSM prior needed).
              Steps 1…: SSM runs forward from z_0 with noise → z_1, …, z_{seg-1}.
              Decoding: autoregressive x̂_t = flow⁻¹(z_t | x̂_{t-1}, cond_t),
                        starting from the patch boundary x_start.
          - Boundary states (seg_starts) computed from x when not supplied.

        Segmented, SSM-prior evaluation (seg_len > 0 and x is None):
          - Each patch samples z_1 ~ N(0, P_inf) and runs SSM from there.
          - seg_starts[s] used as x_prev for the first decoded step.

        Full sequence (seg_len <= 0):
          - One SSM trajectory z_{1:T}, sequential autoregressive decode.
        """
        T    = y.shape[0]
        cond = self._encode(y, u, x0)   # (T, d_out)
        G_L  = self.G_chol_pd
        F    = self.F_stable
        seg  = self.seg_len

        if seg <= 0 or T <= seg:
            # ── Full-sequence: one scan per sample ────────────────────────
            L_inf = jnp.linalg.cholesky(self.P_inf + 1e-6 * jnp.eye(self.d_x))

            def one_sample(k):
                keys_z = jr.split(k, T)
                z1 = L_inf @ jr.normal(keys_z[0], (self.d_x,))
                def ssm_step(z_p, k_i):
                    return (z_t := F @ z_p + G_L @ jr.normal(k_i, (self.d_x,))), z_t
                _, z_rest = jax.lax.scan(ssm_step, z1, keys_z[1:])
                z_all = jnp.concatenate([z1[None], z_rest], axis=0)

                def decode_step(x_p, inputs):
                    zt, ct = inputs
                    xt, _ = _flow_inv_single(self.transforms, zt, x_p, ct)
                    return xt, xt
                _, x_traj = jax.lax.scan(decode_step, x0, (z_all, cond))
                return x_traj

            return jax.vmap(one_sample)(jr.split(key, n_samples))

        # ── Segmented ────────────────────────────────────────────────────
        n_segs    = T // seg
        T_eff     = n_segs * seg
        cond_segs = cond[:T_eff].reshape(n_segs, seg, self.d_out)

        if x is not None:
            # ── Multiple-shooting evaluation: anchor each patch to true x ──
            # Boundary state before patch s: x0 for s=0, x[s*seg-1] for s>0
            x_eff = x[:T_eff]
            boundaries = x_eff[seg - 1 : T_eff - 1 : seg]          # (n_segs-1, d_x)
            seg_starts_ms = jnp.concatenate(
                [x0[None], boundaries], axis=0
            )  # (n_segs, d_x)
            # True first state of each patch: x[0], x[seg], x[2*seg], …
            x_firsts = x_eff.reshape(n_segs, seg, self.d_x)[:, 0, :]  # (n_segs, d_x)

            def one_sample(k):
                seg_keys = jr.split(k, n_segs)

                def sample_and_decode_seg(seg_key, x_start, x_first, cond_s):
                    # Encode true first state → z_0 (deterministic anchor)
                    z0, _ = _flow_fwd_single(
                        self.transforms, x_first, x_start, cond_s[0]
                    )
                    # SSM runs forward from z_0 with noise for the remaining steps
                    step_keys = jr.split(seg_key, seg - 1)
                    def ssm_step(z_p, k_i):
                        z_t = F @ z_p + G_L @ jr.normal(k_i, (self.d_x,))
                        return z_t, z_t
                    _, z_rest = jax.lax.scan(ssm_step, z0, step_keys)
                    z_s = jnp.concatenate([z0[None], z_rest], axis=0)  # (seg, d_x)

                    # Autoregressive decode: x̂_0 ≈ x_first, then propagate
                    def decode_step(x_p, inp):
                        xt, _ = _flow_inv_single(self.transforms, inp[0], x_p, inp[1])
                        return xt, xt
                    _, x_s = jax.lax.scan(decode_step, x_start, (z_s, cond_s))
                    return x_s  # (seg, d_x)

                x_segs_s = jax.vmap(sample_and_decode_seg)(
                    seg_keys, seg_starts_ms, x_firsts, cond_segs
                )
                return x_segs_s.reshape(T_eff, self.d_x)

            return jax.vmap(one_sample)(jr.split(key, n_samples))

        # ── SSM-prior evaluation (x not provided) ────────────────────────
        L_inf = jnp.linalg.cholesky(self.P_inf + 1e-6 * jnp.eye(self.d_x))

        if seg_starts is None:
            seg_starts = jnp.broadcast_to(x0[None], (n_segs, self.d_x))

        def one_sample(k):
            seg_keys = jr.split(k, n_segs)

            def sample_and_decode_seg(seg_key, x_start, cond_s):
                step_keys = jr.split(seg_key, seg)
                z1 = L_inf @ jr.normal(step_keys[0], (self.d_x,))
                def ssm_step(z_p, k_i):
                    return (z_t := F @ z_p + G_L @ jr.normal(k_i, (self.d_x,))), z_t
                _, z_rest = jax.lax.scan(ssm_step, z1, step_keys[1:])
                z_s = jnp.concatenate([z1[None], z_rest], axis=0)

                def decode_step(x_p, inp):
                    xt, _ = _flow_inv_single(self.transforms, inp[0], x_p, inp[1])
                    return xt, xt
                _, x_s = jax.lax.scan(decode_step, x_start, (z_s, cond_s))
                return x_s

            x_segs_s = jax.vmap(sample_and_decode_seg)(seg_keys, seg_starts, cond_segs)
            return x_segs_s.reshape(T_eff, self.d_x)

        return jax.vmap(one_sample)(jr.split(key, n_samples))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_latent_ssm_flow(
    d_x:       int,
    d_u:       int        = 0,
    flow_type: str        = "realnvp",
    seg_len:   int        = 0,
    max_rho:   float      = 0.95,
    key:       jr.PRNGKey = jr.PRNGKey(0),
    **kwargs,
) -> LatentSSMFlow:
    """Factory with sensible defaults.  kwargs forwarded to LatentSSMFlow.__init__."""
    return LatentSSMFlow(
        d_x=d_x, d_u=d_u,
        flow_type=flow_type,
        seg_len=seg_len,
        max_rho=max_rho,
        key=key, **kwargs,
    )