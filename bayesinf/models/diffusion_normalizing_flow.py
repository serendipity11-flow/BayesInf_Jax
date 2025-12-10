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
import optax
from typing import Tuple, List
from networks import TimeSeriesEncoder
from diffusion_model import SinusoidalPosEmb, NoiseNet
from normalizing_flow import ActNorm, RandomPermutation, AffineCoupling, SplineCoupling


class DiffusionNormalizingFlow(eqx.Module):
    """
    Hybrid model combining DDPM and Normalizing Flows for parameter identification.
    
    Architecture:
    - Both DDPM and NF operate directly in parameter space θ
    - DDPM learns the denoising process: θ_T (noise) → θ_0 (clean parameters)
    - NF provides additional structure by mapping θ ↔ z ~ N(0,I)
    
    Training:
    1. DDPM loss: Sample t, apply forward SDE θ → θ_t, learn to predict noise ε
    2. NF loss: Map clean θ through NF to z, maximize log p(z) + log|det J|
    
    Sampling:
    1. Sample z ~ N(0,I) 
    2. Apply NF^(-1): z → θ_init (provides structured initialization)
    3. Apply DDPM reverse: Start from noise, denoise T steps → θ_sample
    
    The NF provides a learnable prior structure, while DDPM handles the
    stochastic generation process conditioned on time series data.
    """
    
    # Encoder for time series conditioning
    encoder: TimeSeriesEncoder
    
    # DDPM components
    ddpm_net: NoiseNet
    time_emb: SinusoidalPosEmb
    
    # Normalizing Flow components
    nf_transforms: List[eqx.Module]
    
    # DDPM noise schedule
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    
    # Model dimensions
    n_steps: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    cond_latent_dim: int = eqx.field(static=True)
    t_transition: float = eqx.field(static=True)
    
    def __init__(
        self,
        feature_dim: int,
        param_dim: int,
        feature_latent_dim: int,  # Encoded condition dimension
        hidden_dim_rnn: int,
        hidden_dim_time: int,
        hidden_sizes_ddpm: List[int],
        hidden_sizes_nf: int,
        num_hidden_nf: int,
        n_steps: int = 1000,
        n_coupling: int = 6,
        flow_type: str = 'realnvp',
        t_transition: float = 0.3,
        *,
        key
    ):
        key_enc, key_ddpm, key_nf = jr.split(key, 3)
        
        self.n_steps = n_steps
        self.param_dim = param_dim
        self.cond_latent_dim = feature_latent_dim
        self.t_transition = t_transition
        
        # Initialize encoder
        self.encoder = TimeSeriesEncoder(
            feature_dim, hidden_dim_rnn, feature_latent_dim, key_enc
        )
        
        # Initialize DDPM components (operates in parameter space θ)
        self.time_emb = SinusoidalPosEmb(dim=hidden_dim_time)
        self.ddpm_net = NoiseNet(
            param_dim,  # DDPM operates directly in parameter space
            feature_latent_dim,
            hidden_dim_time,
            hidden_sizes_ddpm,
            key=key_ddpm
        )
        
        # DDPM noise schedule
        scale = 1000 / n_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = jnp.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        
        # Initialize Normalizing Flow components (operates in parameter space θ)
        self.nf_transforms = []
        keys = jax.random.split(key_nf, n_coupling + 2)
        
        flow_type = flow_type.lower()
        
        def get_mask(parity):
            return jnp.array(
                [(i + parity) % 2 for i in range(param_dim)], 
                dtype=jnp.float32
            )
        
        for i in range(n_coupling // 2):
            k_layer1, k_layer2, k_perm = jax.random.split(keys[i], 3)
            
            # ActNorm for stability
            self.nf_transforms.append(ActNorm(param_dim))
            
            # Random permutation
            self.nf_transforms.append(RandomPermutation(param_dim, key=k_perm))
            
            # Coupling layers based on flow type
            if flow_type == 'nsf':
                self.nf_transforms.append(
                    SplineCoupling(
                        param_dim, 
                        [hidden_sizes_nf] * num_hidden_nf,
                        feature_latent_dim,
                        get_mask(i % 2),
                        key=k_layer1,
                        num_bins=4
                    )
                )
                self.nf_transforms.append(
                    SplineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        feature_latent_dim,
                        get_mask((i + 1) % 2),
                        key=k_layer2,
                        num_bins=4
                    )
                )
            else:  # RealNVP (default)
                self.nf_transforms.append(
                    AffineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        feature_latent_dim,
                        get_mask(i % 2),
                        key=k_layer1
                    )
                )
                self.nf_transforms.append(
                    AffineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        feature_latent_dim,
                        get_mask((i + 1) % 2),
                        key=k_layer2
                    )
                )
        
        # Final ActNorm
        self.nf_transforms.append(ActNorm(param_dim))
    
    def _apply_nf(
        self, 
        h: jnp.ndarray, 
        cond_emb: jnp.ndarray, 
        reverse: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply normalizing flow transformations."""
        logdet = jnp.zeros(h.shape[:-1])
        x = h
        
        flow_list = self.nf_transforms if not reverse else reversed(self.nf_transforms)
        
        for transform in flow_list:
            if isinstance(transform, (ActNorm, RandomPermutation)):
                x, ldj = transform(x, reverse=reverse)
            else:
                x, ldj = transform(x, cond_emb, reverse=reverse)
            logdet = logdet + ldj
        
        return x, logdet
    
    def _theta_to_h_forward_sde(
        self, 
        theta: jnp.ndarray, 
        t: jnp.ndarray,
        key: jax.Array
    ) -> jnp.ndarray:
        """
        Forward SDE process: θ → h_t using DDPM forward diffusion.
        This is the predefined SDE process that adds noise to parameters.
        
        At time t: h_t = sqrt(α_bar_t) * θ + sqrt(1 - α_bar_t) * ε
        """
        eps = jr.normal(key, theta.shape)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        h_t = sqrt_alpha_bar * theta + sqrt_one_minus_alpha_bar * eps
        
        return h_t, eps
    
    def loss(self, x_ts: jnp.ndarray, theta_target: jnp.ndarray, key: jax.Array):
        """
        Combined loss: DDPM loss + NF log-likelihood loss
        
        Training flow:
        1. Sample timestep t
        2. Apply forward SDE: θ → h_t (add noise)
        3. Learn to denoise: predict ε from h_t
        4. Apply NF: h_0 → z, compute log-likelihood
        """
        batch_size = theta_target.shape[0]
        key_t, key_eps, key_nf = jr.split(key, 3)
        
        # Encode condition
        cond = self.encoder(x_ts)
        
        # === DDPM Loss (θ space) ===
        # Sample random timesteps
        t = jr.randint(key_t, (batch_size,), minval=0, maxval=int(self.n_steps * self.t_transition))
        
        # Forward SDE: θ → h_t (predefined noise schedule)
        h_t, eps = self._theta_to_h_forward_sde(theta_target, t, key_eps)
        
        # Time embedding
        t_emb = self.time_emb(t)

        
        # Predict noise from h_t
        eps_pred = jax.vmap(self.ddpm_net)(h_t, t_emb, cond)
        ddpm_loss = jnp.mean((eps - eps_pred) ** 2)
        
        # === Normalizing Flow Loss ===
        # For NF loss, we use the next step of h_t to z
        h_t_next, _ = self._theta_to_h_forward_sde(theta_target, t+1, key_eps)
        z, logdet = self._apply_nf(h_t_next, cond, reverse=False)
        
        # Prior log probability N(0, I)
        log_prob_z = -0.5 * (
            jnp.sum(z**2, axis=-1) + self.param_dim * jnp.log(2 * jnp.pi)
        )
        
        # NF loss is negative log-likelihood
        nf_loss = -jnp.mean(log_prob_z + logdet)
        
        # Combined loss (you can adjust the weight)
        total_loss = ddpm_loss + 0.1 * nf_loss
        
        return total_loss
    
    @eqx.filter_jit
    def sample(
        self, 
        x_ts: jnp.ndarray, 
        key: jax.Array, 
        num_samples: int = 1
    ) -> jnp.ndarray:
        """
        Two-stage sampling:
        1. Sample z ~ N(0,I) and transform through NF^(-1) to get h
        2. Run DDPM reverse process: h → θ (denoising)
        
        The key insight: h from NF^(-1) is our starting "clean" latent,
        which we then noisify and denoise through DDPM to get θ samples
        """
        batch_size = x_ts.shape[0]
        cond = self.encoder(x_ts)
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        total_num = batch_size * num_samples
        
        # === Stage 1: Sample from NF ===
        key, k_z = jr.split(key)
        z_init = jr.normal(k_z, (total_num, self.param_dim))
        
        # Transform z through inverse NF to get h (intermediate latent)
        h_clean, _ = self._apply_nf(z_init, cond_rep, reverse=True)
        
        # === Stage 2: DDPM Reverse Process ===
        # Start from pure noise and denoise toward h_clean
        # We start from h_T (fully noised) and denoise to h_0 = θ
        key, k_init = jr.split(key)
        
        def step_fn(val, i):
            h, key = val
            t_idx = int(self.n_steps*self.t_transition) - 1 - i
            
            t_batch = jnp.full((total_num,), t_idx, dtype=jnp.int32)
            t_emb = self.time_emb(t_batch)

            # Predict noise
            eps_pred = jax.vmap(self.ddpm_net)(h, t_emb, cond_rep)
            
            # DDPM reverse step coefficients
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            
            # Posterior mean: μ_θ(h_t, t)
            mean = (1 / jnp.sqrt(alpha)) * (
                h - (beta / jnp.sqrt(1 - alpha_bar)) * eps_pred
            )
            
            # Add noise (except at final step)
            key, k_noise = jr.split(key)
            noise = jr.normal(k_noise, h.shape)
            sigma = jnp.sqrt(beta)
            
            h_next = mean + jnp.where(t_idx > 0, sigma * noise, 0.0)
            return (h_next, key), None
        
        # Run denoising process
        (theta_samples, _), _ = jax.lax.scan(
            step_fn, (h_clean, key), jnp.arange(int(self.n_steps*self.t_transition))
        )
        
        # Reshape to (num_samples, batch_size, param_dim)
        samples = theta_samples.reshape(batch_size, num_samples, self.param_dim)
        return jnp.swapaxes(samples, 0, 1)


@eqx.filter_jit
def make_step(
    model: DiffusionNormalizingFlow,
    x_ts: jnp.ndarray,
    theta: jnp.ndarray,
    key: jax.Array,
    opt_state,
    optimizer: optax.GradientTransformation
):
    """Training step for Diffusion Normalizing Flow."""
    loss_fn = lambda m: m.loss(x_ts, theta, key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss