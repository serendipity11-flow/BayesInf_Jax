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
from functools import partial
from tqdm import tqdm
from networks import TimeSeriesEncoder
import numpy as np


class VectorField(eqx.Module):
    """
    Approximates the time-dependent vector field v_t(z|cond).
    Input: t (scalar), z (param_dim), cond (latent_dim)
    Output: dz/dt (param_dim)
    """
    net: eqx.nn.MLP

    def __init__(self, param_dim, cond_dim, hidden_dim, depth, key):
        # Input size: time(1) + z(param_dim) + cond(cond_dim)
        input_dim = 1 + param_dim + cond_dim
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=param_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.gelu,  # GELU usually works better for Flows
            key=key
        )

    def __call__(self, t, z, cond):
        t_arr = jnp.array([t])
        x = jnp.concatenate([t_arr, z, cond])
        return self.net(x)


class ConditionalFlowMatching(eqx.Module):
    encoder: TimeSeriesEncoder
    vector_field: VectorField
    param_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, feature_dim, param_dim, latent_dim, hidden_sizes, hidden_size_rnn, key):
        key_enc, key_vf = jr.split(key)
        self.param_dim = param_dim
        self.latent_dim = latent_dim

        self.encoder = TimeSeriesEncoder(feature_dim, hidden_size_rnn, latent_dim, key_enc)
        self.vector_field = VectorField(param_dim, latent_dim, hidden_sizes[0], depth=len(hidden_sizes), key=key_vf)

    def __call__(self, t, z, x_ts):
        cond = self.encoder(x_ts)
        return jax.vmap(self.vector_field)(t, z, cond)

    def sample(self, x_ts, key, num_samples=1, steps=100):

        cond = self.encoder(x_ts)
        batch_size = x_ts.shape[0]
        total_num = batch_size * num_samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)

        z0 = jr.normal(key, (total_num, self.param_dim))

        # Simple Euler Integration (or RK4)
        # Since we use OT path, the trajectory is straight, so Euler is often sufficient.
        dt = 1.0 / steps

        def body_fn(i, z_curr):
            t = i * dt
            v = jax.vmap(self.vector_field, in_axes=(None, 0, 0))(t, z_curr, cond_rep)
            return z_curr + v * dt

        z1 = jax.lax.fori_loop(0, steps, body_fn, z0)

        samples = z1.reshape(batch_size, num_samples, self.param_dim)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples


def cfm_loss(model, x_ts_batch, theta_batch, key):
    B = theta_batch.shape[0]
    key_t, key_z0 = jr.split(key)
    t = jr.uniform(key_t, (B,))
    z0 = jr.normal(key_z0, theta_batch.shape)
    z1 = theta_batch
    t_b = t[:, None]
    z_t = (1 - t_b) * z0 + t_b * z1
    v_target = z1 - z0
    v_pred = model(t, z_t, x_ts_batch)
    return jnp.mean((v_pred - v_target) ** 2)


@eqx.filter_jit
def make_step(model, x_ts, theta, key, opt_state, optimizer):
    loss_fn = lambda m: cfm_loss(m, x_ts, theta, key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss