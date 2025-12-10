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
from typing import Any, Literal, List
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
from utils.common import gaussian_kl_divergence, reparameterize
from networks import TimeSeriesEncoder


class cVAE(eqx.Module):
    """
    Conditional VAE for parameter identification from time series observations.

    Use case: Given observed time series, infer posterior distribution over system parameters.
    Example: Given trajectory observations → estimate pendulum mass/length/damping
    This performs Bayesian inference for inverse problems.
    """

    ts_encoder: TimeSeriesEncoder
    p_encoder: eqx.Module
    l_mu: eqx.nn.Linear
    l_logvar: eqx.nn.Linear
    p_decoder: eqx.Module
    latent_dim: int

    def __init__(self, feature_dim: int, param_dim: int, latent_dim: int = 16, hidden_sizes: List[int] = [64]*2,
                 hidden_dim_rnn: int = 64, rnn_type: Literal["lstm", "gru"] = "lstm", *, key):
        """
        Initialize parameter identification CVAE.

        Args:
            feature_dim: Dimension of features at each timestep
            param_dim: Dimension of parameters to identify
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for vae encoder/decoder
            hidden_dim_rnn: Hidden dimensions for RNN layer
            rnn_type: Type of RNN ("lstm" or "gru")
            key: JAX random key
        """
        ts_key, enc_key, l_mu, l_logvar, dec_key = jr.split(key, 5)

        self.latent_dim = latent_dim
        self.ts_encoder = TimeSeriesEncoder(feature_dim, hidden_dim_rnn, latent_dim, ts_key, rnn=rnn_type)
        self.p_encoder = eqx.nn.MLP(param_dim + latent_dim, latent_dim, hidden_sizes[0], depth=len(hidden_sizes), key=enc_key)
        self.l_mu = eqx.nn.Linear(latent_dim, latent_dim, key=l_mu)
        self.l_logvar = eqx.nn.Linear(latent_dim, latent_dim, key=l_logvar)
        self.p_decoder = eqx.nn.MLP(latent_dim * 2, param_dim, hidden_sizes[0], depth=len(hidden_sizes), key=dec_key)

    def __vae_call__(self, emb: jnp.ndarray, p: jnp.ndarray, key):
        enc = self.p_encoder(jnp.concatenate([p, emb], axis=-1))
        mean = self.l_mu(enc)
        logvar = self.l_logvar(enc)
        z = reparameterize(key, mean, logvar)
        params_recon = self.p_decoder(jnp.concatenate([z, emb]))
        return params_recon, mean, logvar

    def __call__(self, x: jnp.ndarray, p: jnp.ndarray, key):
        enc_ts = self.ts_encoder(x)
        keys = jr.split(key, x.shape[0])
        params_recon, mean, logvar = jax.vmap(self.__vae_call__)(enc_ts, p, keys)
        return params_recon, mean, logvar

    def sample(self, x: jnp.ndarray, key: jax.random.PRNGKey, num_samples: int = 100):
        enc_ts = self.ts_encoder(x)
        batch_size = x.shape[0]
        total_num = batch_size * num_samples
        enc_ts_rep = jnp.repeat(enc_ts, num_samples, axis=0)

        z = jr.normal(key, (total_num, self.latent_dim,))
        params_recon = jax.vmap(self.p_decoder)(jnp.concatenate([z, enc_ts_rep], axis=-1))
        samples = params_recon.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples


def cvae_loss(model, x: jnp.ndarray, true_params: jnp.ndarray, key: jax.random.PRNGKey, beta: float = 1.0):
    params_recon, mean, logvar = model(x, true_params, key)

    # Reconstruction loss (MSE over parameters)
    recon_loss = jnp.mean(jnp.sum((true_params - params_recon) ** 2, axis=-1))

    # KL divergence
    kl_loss = jnp.mean(gaussian_kl_divergence(mean, logvar))

    loss = recon_loss + beta * kl_loss

    return loss


@eqx.filter_jit
def make_step(model, x: jnp.ndarray, true_params: jnp.ndarray, key: jax.random.PRNGKey, opt_state: Any,
              optimizer: optax.GradientTransformation, beta: float = 1.0):
    """Training step for parameter identification CVAE."""
    loss_fn = lambda m: cvae_loss(m, x, true_params, key, beta)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss