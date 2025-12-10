import os
from numba import cuda

if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax

    jax.default_device('cpu')
from typing import List
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm
from functools import partial
from networks import TimeSeriesEncoder
import numpy as np


class VariationalLinear(eqx.Module):
    """
    A Linear layer where weights/biases are random variables with Normal posterior.
    Uses Softplus for scale parametrization for numerical stability.
    """
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    weight_mu: jax.Array
    weight_rho: jax.Array  # rho is the unconstrained parameter for sigma
    bias_mu: jax.Array
    bias_rho: jax.Array

    prior_mean: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, key: jr.PRNGKey, prior_std: float = 1.0):
        key_w, key_b = jr.split(key, 2)
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = 0.0
        self.prior_std = prior_std

        # Initialization
        limit = jnp.sqrt(1.0 / in_features)
        self.weight_mu = jr.uniform(key_w, (out_features, in_features), minval=-limit, maxval=limit)
        self.bias_mu = jnp.zeros((out_features,))

        # Initialize rho such that softplus(rho) approx 0.05 (small initial variance)
        # softplus(-3.0) ~ 0.048
        self.weight_rho = jnp.full((out_features, in_features), -3.0)
        self.bias_rho = jnp.full((out_features,), -3.0)

    def sample_weights(self, key: jr.PRNGKey):
        key_w, key_b = jr.split(key, 2)

        # Softplus for stability: sigma = log(1 + exp(rho))
        w_sigma = jax.nn.softplus(self.weight_rho)
        b_sigma = jax.nn.softplus(self.bias_rho)

        w_eps = jr.normal(key_w, self.weight_mu.shape)
        b_eps = jr.normal(key_b, self.bias_mu.shape)

        W = self.weight_mu + w_sigma * w_eps
        b = self.bias_mu + b_sigma * b_eps
        return W, b

    def kl_divergence(self):
        w_sigma = jax.nn.softplus(self.weight_rho)
        b_sigma = jax.nn.softplus(self.bias_rho)
        prior_var = self.prior_std ** 2

        # KL(N(mu, sigma^2) || N(0, prior_std^2))
        # 0.5 * (var/prior_var + mu^2/prior_var - 1 - log(var/prior_var))

        t1_w = (w_sigma ** 2 + self.weight_mu ** 2) / prior_var
        t2_w = -1.0 - 2.0 * (jnp.log(w_sigma) - jnp.log(self.prior_std))
        kl_w = 0.5 * jnp.sum(t1_w + t2_w)

        t1_b = (b_sigma ** 2 + self.bias_mu ** 2) / prior_var
        t2_b = -1.0 - 2.0 * (jnp.log(b_sigma) - jnp.log(self.prior_std))
        kl_b = 0.5 * jnp.sum(t1_b + t2_b)

        return kl_w + kl_b

    def __call__(self, x: jax.Array, key: jr.PRNGKey):
        W, b = self.sample_weights(key)
        return jnp.dot(W, x) + b


class BNNPosteriorEstimator(eqx.Module):
    """
    The full model: GRU Encoder + BNN Inference Head.
    It learns to map Time Series D to the parameters (mu, log_sigma) of the physical parameters theta_p.
    """
    encoder: TimeSeriesEncoder
    bnn_layers: list[VariationalLinear]

    # Static info about physical parameters (theta_p)
    param_dim: int = eqx.field(static=True)

    def __init__(self, feature_dim: int, param_dim: int, latent_dim: int, hidden_sizes: List[int], hidden_size_rnn: int, 
    key: jr.PRNGKey):

        key_enc, key = jr.split(key)
        self.encoder = TimeSeriesEncoder(feature_dim, hidden_size_rnn, latent_dim, key_enc)
        self.param_dim = param_dim
        self.bnn_layers = []
        keys = jr.split(key, len(hidden_sizes)+1)
        for i in range(len(hidden_sizes)):
            self.bnn_layers += [VariationalLinear(latent_dim, hidden_sizes[i], key=keys[i]),
                                eqx.nn.Lambda(jax.nn.gelu)]
        self.bnn_layers.append(VariationalLinear(hidden_sizes[-1], 2 * param_dim, key=keys[-1]))

    def __bnn_call__(self, emb: jax.Array, model_key: jr.PRNGKey):
        """
        emb: embedded Time series observation (Features)
        key: PRNG key for sampling the BNN weights (theta_NN)
        Returns: mu, log_sigma for the physical parameters (theta_p)
        """
        key, _ = jr.split(model_key, 2)
        key_layers = jr.split(key, len(self.bnn_layers))
        x = emb
        for i, layer in enumerate(self.bnn_layers):
            if isinstance(layer, VariationalLinear):
                x = layer(x, key_layers[i])
            else:  # Activation
                x = layer(x)

        # Split the output into mu and log_sigma for physical parameters theta_p
        mu_theta_p = x[:self.param_dim]
        log_sigma_theta_p = x[self.param_dim:]

        return mu_theta_p, log_sigma_theta_p

    def __call__(self, D: jax.Array, key: jr.PRNGKey):
        emb_D = self.encoder(D)
        mu_theta_p, log_sigma_theta_p = jax.vmap(self.__bnn_call__, in_axes=(0, None))(emb_D, key)
        return mu_theta_p, log_sigma_theta_p

    def kl_divergence(self):
        """Calculates the total KL divergence for all VariationalLinear layers (theta_NN)."""
        kl_sum = 0.0
        # Iterate over all layers that have a kl_divergence method
        for layer in self.bnn_layers:
            if isinstance(layer, VariationalLinear):
                kl_sum += layer.kl_divergence()
        return kl_sum

    def sample(self, D: jax.Array, key, n_samples: int):
        key, noise_key = jr.split(key)
        # each weight should be sampled individually
        # D: batch_size, time_length, D_in
        # return: num_samples * batch_size * num_theta
        emb_D = self.encoder(D)
        sample_fn = lambda d, k: self.__bnn_call__(d, k)
        def batched_sample_fn(k_carry, _):
            k, nk = k_carry
            k, k1 = jr.split(k)
            nk, _ = jr.split(nk)
            mus, log_sigmas = jax.vmap(sample_fn, axis_name="batch")(emb_D, jr.split(k1, D.shape[0]))
            sigmas = jnp.exp(log_sigmas)
            eps = jr.normal(nk, sigmas.shape)
            return (k, nk), mus + sigmas * eps
        _, samples = jax.lax.scan(batched_sample_fn, (key, noise_key), length=n_samples)
        return samples


def log_likelihood(model: BNNPosteriorEstimator, D_i: jax.Array, theta_p_true_i: jax.Array, key: jr.PRNGKey):
    # Note: We pass the key here to sample the BNN weights for this single example
    mu_theta_p, log_sigma_theta_p = model(D_i, key)
    sigma_theta_p = jnp.exp(log_sigma_theta_p)

    # 2. Log-Likelihood Term
    # Log P(theta_p_true_i | D_i) for a Gaussian likelihood
    log_likelihood = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * sigma_theta_p ** 2)
                             - 0.5 * ((theta_p_true_i - mu_theta_p) / sigma_theta_p) ** 2)

    return log_likelihood


def batched_elbo_loss(model: BNNPosteriorEstimator,
                      D_batch: jax.Array,
                      theta_p_true_batch: jax.Array,
                      key: jr.PRNGKey,
                      kl_weight=1e-3):
    """
    Computes the Evidence Lower Bound (ELBO) loss for a BATCH of data.
    """
    key, _ = jr.split(key, 2)
    log_likelihoods = log_likelihood(model, D_batch, theta_p_true_batch, key)
    kl_divergence = model.kl_divergence()
    loss = kl_weight * kl_divergence.mean() - log_likelihoods.mean()
    return loss


def make_kl_annealing_schedule(
        start_step: int,
        end_step: int,
        final_kl_weight: float
):
    """
    Creates a Sigmoidal Annealing schedule for the KL weight using Optax.

    The weight increases from 0 to final_kl_weight over the annealing period.

    Args:
        start_step: The global step where annealing begins.
        end_step: The global step where the final weight is reached.
        final_kl_weight: The maximum KL weight to reach.

    Returns:
        A function (step) -> kl_weight.
    """

    def sigmoid_fn(step):
        # Calculate the progress ratio (0 to 1)
        progress = (step - start_step) / (end_step - start_step)
        progress = jnp.clip(progress, 0.0, 1.0)

        # Sigmoidal function: 1 / (1 + exp(-x)). We scale x so the curve works.
        # We want the output to go from 0 to 1 smoothly.
        # A common sigmoid-based ramp uses tanh or a shifted sigmoid.
        # Here we'll use a simple linear ramp for maximum control/predictability:
        # return progress # Use for Linear Annealing

        # Using a smooth sigmoid-based ramp (e.g., tanh or double sigmoid)
        # We map progress [0, 1] to [-C, C] where C defines the steepness. C=6 is standard.
        steepness = 6.0
        x = steepness * (2 * progress - 1)

        # Calculate a smooth ramp from 0 to 1
        smooth_ramp = 1.0 / (1.0 + jnp.exp(-x))

        # Map the ramp's [~0, ~1] range to a precise [0, 1] range:
        # smooth_ramp is approx 1/(1+exp(steepness)) at progress=0, and 1 at progress=1.
        # For steepness=6, this is (0.0024, 0.9975). We will just clip.
        ramp = jnp.clip(smooth_ramp, 0.0, 1.0)

        return final_kl_weight * ramp

    return sigmoid_fn

@eqx.filter_jit
def make_step(model, D_batch, theta_p_true_batch, key, opt_state, optimizer, kl_weight=1e-3):
    """Performs one step of ELBO minimization (gradient descent) with a batch."""

    # Use filter_value_and_grad on the batched_elbo_loss
    loss, grads = eqx.filter_value_and_grad(batched_elbo_loss)(
        model, D_batch, theta_p_true_batch, key, kl_weight=kl_weight
    )

    # Update the model and optimizer state
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss