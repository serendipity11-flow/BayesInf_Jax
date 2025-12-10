import os
from numba import cuda

if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax

    jax.default_device('cpu')
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from networks import TimeSeriesEncoder
import math


class SinusoidalPosEmb(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        # x: (batch_size, ) or scalar
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class FiLM(eqx.Module):
    """Feature-wise Linear Modulation block."""
    gamma: eqx.nn.Linear
    beta: eqx.nn.Linear

    def __init__(self, width, key):
        k1, k2 = jr.split(key)
        self.gamma = eqx.nn.Linear(width, width, key=k1)
        self.beta = eqx.nn.Linear(width, width, key=k2)

    def __call__(self, h, cond):
        """cond should be the embedding controlling modulation."""
        return self.gamma(cond) * h + self.beta(cond)


class CrossAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    proj: eqx.nn.Linear
    dim: int

    def __init__(self, dim, key):
        self.dim = dim
        kq, kk, kv, kp = jr.split(key, 4)
        self.query = eqx.nn.Linear(dim, dim, key=kq)
        self.key = eqx.nn.Linear(dim, dim, key=kk)
        self.value = eqx.nn.Linear(dim, dim, key=kv)
        self.proj = eqx.nn.Linear(dim, dim, key=kp)

    def __call__(self, q_emb, kv_emb):
        # Single-query cross-attention
        Q = self.query(q_emb)        # (B, dim)
        K = self.key(kv_emb)         # (B, dim)
        V = self.value(kv_emb)       # (B, dim)

        attn = jax.nn.softmax(jnp.sum(Q * K, axis=-1, keepdims=True) / jnp.sqrt(self.dim), axis=0)
        out = attn * V
        return q_emb + self.proj(out)  # Residual
        

class NoiseNet(eqx.Module):
    time_mlp: eqx.nn.Sequential
    input_proj: eqx.nn.Linear
    blocks: list
    film_layers: list
    attn: CrossAttention
    output_proj: eqx.nn.Linear

    def __init__(self, param_dim, cond_dim, time_dim, hidden_sizes, key):
        k_time, k_in, k_out, k_attn, *keys = jr.split(key, 4 + len(hidden_sizes)*2)
        width = hidden_sizes[0]

        # Time embedding MLP
        self.time_mlp = eqx.nn.Sequential([
            eqx.nn.Linear(time_dim, width, key=k_time),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(width, width, key=jr.fold_in(k_time, 1)),
        ])

        # Initial projection
        in_dim = param_dim + width + cond_dim
        self.input_proj = eqx.nn.Linear(in_dim, width, key=k_in)

        # Cross-attention to fuse cond into the flow
        self.attn = CrossAttention(width, key=k_attn)

        # MLP blocks with FiLM conditioning
        self.blocks = []
        self.film_layers = []
        for i in range(len(hidden_sizes)):
            k1, k2 = jr.split(keys[2*i])
            self.blocks.append(eqx.nn.Sequential([
                eqx.nn.Linear(width, width, key=k1),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(width, width, key=k2),
            ]))
            self.film_layers.append(FiLM(width, keys[2*i+1]))

        # Output
        self.output_proj = eqx.nn.Linear(width, param_dim, key=k_out)

    def __call__(self, z_t, t_emb, cond):
        t_cond = self.time_mlp(t_emb)  # time embedding processed

        # Cross-attention injection of cond into z
        h0 = jnp.concatenate([z_t, t_cond, cond], axis=-1)
        h = self.input_proj(h0)
        h = self.attn(h, cond)  # stronger conditioning

        # MLP + FiLM blocks
        for block, film in zip(self.blocks, self.film_layers):
            h_res = block(h)
            h = h + film(h_res, t_cond)

        return self.output_proj(h)

class ConditionalDDPM(eqx.Module):
    encoder: TimeSeriesEncoder
    net: NoiseNet
    time_emb: SinusoidalPosEmb
    
    # Static arrays
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    
    n_steps: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    t_cond_start: float = eqx.field(static=True)

    def __init__(self, feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time, hidden_sizes_noise, n_steps=1000, *, key, t_cond_start):
        key_enc, key_net = jr.split(key)
        self.n_steps = n_steps
        self.param_dim = param_dim
        self.t_cond_start = t_cond_start

        self.encoder = TimeSeriesEncoder(feature_dim, hidden_dim_rnn, feature_latent_dim, key_enc)
        self.time_emb = SinusoidalPosEmb(dim=hidden_dim_time)
        self.net = NoiseNet(param_dim, feature_latent_dim, hidden_dim_time, hidden_sizes_noise, key=key_net)

        # Linear Noise Schedule
        scale = 1000 / n_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02 # Slightly increased max beta for better coverage
        self.betas = jnp.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def loss(self, x_ts, theta_target, key):
        batch_size = theta_target.shape[0]
        key_t, key_eps = jr.split(key)
        
        t = jr.randint(key_t, (batch_size,), minval=0, maxval=self.n_steps)
        eps = jr.normal(key_eps, theta_target.shape)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        z_t = sqrt_alpha_bar * theta_target + sqrt_one_minus_alpha_bar * eps
        
        cond = self.encoder(x_ts)
        
        t_emb = self.time_emb(t)

        gate = (t <= self.t_cond_start * self.n_steps).astype(jnp.float32)[:, None]
        cond_input = cond * gate

        eps_pred = jax.vmap(self.net)(z_t, t_emb, cond_input)
        return jnp.mean((eps - eps_pred) ** 2)

    @eqx.filter_jit
    def sample(self, x_ts, key, num_samples=1):
        """Ancestral Sampling (JIT Compiled)"""
        batch_size = x_ts.shape[0]
        cond = self.encoder(x_ts)
        # Repeat condition for num_samples: (B, Latent) -> (B * N, Latent)
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        total_num = batch_size * num_samples

        key, k_init = jr.split(key)
        z_T = jr.normal(k_init, (total_num, self.param_dim))

        def step_fn(val, i):
            z, key = val
            t_idx = self.n_steps - 1 - i
            
            t_batch = jnp.full((total_num,), t_idx, dtype=jnp.int32)
            t_emb = self.time_emb(t_batch)
            
            cond_input = jax.lax.cond(t_idx < self.t_cond_start * self.n_steps, 
            lambda _: cond_rep, lambda _: jnp.zeros_like(cond_rep), operand=None)
            eps_pred = jax.vmap(self.net)(z, t_emb, cond_input)
            
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            
            # DDPM Update
            mean = (1 / jnp.sqrt(alpha)) * (z - (beta / jnp.sqrt(1 - alpha_bar)) * eps_pred)
            
            key, k_noise = jr.split(key)
            noise = jr.normal(k_noise, z.shape)
            sigma = jnp.sqrt(beta)
            
            z_next = mean + jnp.where(t_idx > 0, sigma * noise, 0.0)
            return (z_next, key), None # Scan requires return (carrier, output)

        # Use Scan for faster compilation loop
        (z_0, _), _ = jax.lax.scan(step_fn, (z_T, key), jnp.arange(self.n_steps))
        
        samples = z_0.reshape(batch_size, num_samples, self.param_dim)
        return jnp.swapaxes(samples, 0, 1)


class ConditionalScoreModel(eqx.Module):
    encoder: TimeSeriesEncoder
    net: NoiseNet
    time_emb: SinusoidalPosEmb
    param_dim: int = eqx.field(static=True)
    beta_min: float = 0.1
    beta_max: float = 20.0

    def __init__(self, feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time, hidden_sizes_noise, *, key):
        key_enc, key_net = jr.split(key)
        self.param_dim = param_dim
        self.encoder = TimeSeriesEncoder(feature_dim, hidden_dim_rnn, feature_latent_dim, key_enc)
        self.time_emb = SinusoidalPosEmb(dim=hidden_dim_time)
        self.net = NoiseNet(param_dim, feature_latent_dim, hidden_dim_time, hidden_sizes_noise, key=key_net)

    def get_marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = jnp.expand_dims(log_mean_coeff, axis=-1)
        mean = x * jnp.exp(log_mean_coeff)
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def loss(self, x_ts, theta_target, key):
        batch_size = theta_target.shape[0]
        key_t, key_z = jr.split(key)
        t = jr.uniform(key_t, (batch_size, ), minval=1e-4, maxval=1.0)
        
        noise = jr.normal(key_z, theta_target.shape)
        mean, std = self.get_marginal_prob(theta_target, t)
        z_t = mean + std * noise
        
        cond = self.encoder(x_ts)
        
        t_emb = self.time_emb(t)
        
        # Predict: score * std (Denoising Score Matching)
        score_pred_scaled = jax.vmap(self.net)(z_t, t_emb, cond)
        
        # Target: -noise (Standard Gaussian DSM objective)
        return jnp.mean((score_pred_scaled - (-noise)) ** 2)

    @eqx.filter_jit
    def sample(self, x_ts, key, num_samples=1, steps=1000):
        """
        Standard Reverse SDE (Euler-Maruyama)
        dx = [f(x,t) - g(t)^2 score(x,t)] dt + g(t) dw
        """
        batch_size = x_ts.shape[0]
        cond = self.encoder(x_ts)
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        total_samples = batch_size * num_samples

        key, k_init = jr.split(key)
        z_init = jr.normal(k_init, (total_samples, self.param_dim))
        
        dt = -1.0 / steps 

        def step_fn(val, i):
            z, key = val
            t = 1.0 - (i / steps)
            t_batch = jnp.full((total_samples,), t)
            t_emb = self.time_emb(t_batch)

            # SDE Coefficients
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            drift_fwd = -0.5 * beta_t * z
            diffusion_g = jnp.sqrt(beta_t)
            
            # Predict Score
            _, std_t = self.get_marginal_prob(jnp.ones_like(z), t)
            output_scaled = jax.vmap(self.net)(z, t_emb, cond_rep)
            
            # Unscale score: output approx -epsilon, so score approx -epsilon/std
            score = output_scaled / (std_t + 1e-6)

            # Reverse SDE Drift: f - g^2 * score
            # (Note: Factor is 1.0 for g^2 in SDE, 0.5 is for ODE)
            rev_drift = drift_fwd - (diffusion_g ** 2) * score
            
            key, k_step = jr.split(key)
            dw = jr.normal(k_step, z.shape) * jnp.sqrt(jnp.abs(dt))
            
            z_next = z + rev_drift * dt + diffusion_g * dw
            
            # Safety Clip to prevent explosion during early training
            z_next = jnp.clip(z_next, -5.0, 5.0) 
            
            return (z_next, key), None

        (z_final, _), _ = jax.lax.scan(step_fn, (z_init, key), jnp.arange(steps))
        
        samples = z_final.reshape(batch_size, num_samples, self.param_dim)
        return jnp.swapaxes(samples, 0, 1)

@eqx.filter_jit
def make_step(model, x_ts, theta, key, opt_state, optimizer):
    loss_fn = lambda m: m.loss(x_ts, theta, key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss