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
from bayesinf.models.networks import TimeSeriesEncoder
from typing import Optional
from bayesinf.models.MCMC_Flow import MALALayer, RMALALayer, MCMCLayer, BMCMCLayer
STOCHASTIC_LAYERS = (MALALayer, RMALALayer, MCMCLayer, BMCMCLayer)


class SinusoidalPosEmb(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t):
        """
        t: array-like of shape (batch,) or scalar ints/floats
        returns: (batch, dim) float32
        """
        # Ensure float array
        t = jnp.asarray(t).astype(jnp.float32)
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        # log(10000) scale
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / float(max(1, half - 1)))
        # outer product: (batch, half)
        args = t[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        # If dim is odd, pad one zero column
        if emb.shape[-1] < self.dim:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb.astype(jnp.float32)

class ResBlock(eqx.Module):
    """Helper Residual Block for better gradient flow"""
    linear: eqx.nn.Linear
    act: callable

    def __init__(self, dim, key):
        self.linear = eqx.nn.Linear(dim, dim, key=key)
        self.act = jax.nn.gelu

    def __call__(self, x):
        return x + self.act(self.linear(x))

class NoiseNet(eqx.Module):
    """
    Upgraded Network with Residual connections.
    Better at fusing Condition (c) with Noisy State (z_t).
    """
    time_mlp: eqx.nn.Sequential
    input_proj: eqx.nn.Linear
    res_blocks: list
    output_proj: eqx.nn.Linear

    def __init__(self, param_dim, cond_dim, time_dim, hidden_sizes, key):
        k_time, k_in, k_out, *k_res = jr.split(key, 3 + len(hidden_sizes))
        
        # Time Embedding Projection
        self.time_mlp = eqx.nn.Sequential([
            eqx.nn.Linear(time_dim, time_dim * 4, key=k_time),
            eqx.nn.Lambda(jax.nn.swish),
            eqx.nn.Linear(time_dim * 4, time_dim, key=jr.fold_in(k_time, 1))
        ])

        # Input Projection
        in_dim = param_dim + time_dim + cond_dim
        width = hidden_sizes[0]
        self.input_proj = eqx.nn.Linear(in_dim, width, key=k_in)

        # Residual Blocks
        self.res_blocks = [ResBlock(width, k) for k in k_res]

        # Output Projection
        self.output_proj = eqx.nn.Linear(width, param_dim, key=k_out)

    def __call__(self, z_t, t_emb, cond):
        t_feat = self.time_mlp(t_emb)
        # Concatenate inputs
        x = jnp.concatenate([z_t, t_feat, cond], axis=-1)
        
        # Project and Residual Process
        h = jax.nn.swish(self.input_proj(x))
        for block in self.res_blocks:
            h = block(h)
            
        return self.output_proj(h)

class ConditionalScoreModel(eqx.Module):
    encoder: TimeSeriesEncoder
    net: NoiseNet
    time_emb: SinusoidalPosEmb
    param_dim: int = eqx.field(static=True)
    beta_min: float = 0.1
    beta_max: float = 20.0
    refinement: Optional[eqx.Module] 

    def __init__(self, feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time, hidden_sizes_noise, *, key,
                 log_posterior_fn=None, mcmc_type='MALA', num_mh_steps: int = 1, mcmc_step_size: float = 1e-3, noise_std=0.01):
        key_enc, key_net = jr.split(key)
        self.param_dim = param_dim
        self.encoder = TimeSeriesEncoder(feature_dim, hidden_dim_rnn, feature_latent_dim, key_enc)
        self.time_emb = SinusoidalPosEmb(dim=hidden_dim_time)
        self.net = NoiseNet(param_dim, feature_latent_dim, hidden_dim_time, hidden_sizes_noise, key=key_net)
        if log_posterior_fn is not None:
            if mcmc_type.upper() == 'MALA':
                self.refinement = MALALayer(log_posterior_fn, 
                        lambd=1.0, num_langevin_steps=1, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                        apply_gradient_schedule=False
                        )                       
            elif mcmc_type.upper() == 'RMALA':
                self.refinement = RMALALayer(log_posterior_fn, 
                    lambd=1.0, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                    apply_gradient_schedule=False
                    )
            elif mcmc_type.upper() == 'BMCMC':
                self.refinement = BMCMCLayer(log_posterior_fn, 
                    lambd=1.0, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                    apply_gradient_schedule=False
                    )
            else:
                self.refinement = MCMCLayer(log_posterior_fn, 
                    lambd=1.0, num_mh_steps=num_mh_steps, noise_std=noise_std, 
                    apply_gradient_schedule=False
                    )
        else:
            self.refinement = None

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
    def sample_with_MCMC(self, x_ts, key, num_samples=1, steps=1000):
        """
        Standard Reverse SDE (Euler-Maruyama)
        dx = [f(x,t) - g(t)^2 score(x,t)] dt + g(t) dw
        """
        batch_size = x_ts.shape[0]
        cond_emb = self.encoder(x_ts)
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)
        cond_rep = jnp.repeat(x_ts, num_samples, axis=0)
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
            output_scaled = jax.vmap(self.net)(z, t_emb, cond_emb_rep)
            
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
        if self.refinement:
            s_key = jr.split(key, 2)[1]
            z_final = self.refinement(z_final, cond_rep, s_key, cond_emb_rep)[0]
        
        samples = z_final.reshape(batch_size, num_samples, self.param_dim)
        return jnp.swapaxes(samples, 0, 1)

@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state, optimizer, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return full_model.loss(y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
    else:
        loss_fn = lambda m: m.loss(y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
    return model, opt_state, loss