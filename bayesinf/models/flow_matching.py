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
    refinement: Optional[eqx.Module] 

    def __init__(self, feature_dim, param_dim, latent_dim, hidden_sizes, hidden_size_rnn, key, log_posterior_fn=None,
                 mcmc_type='MALA', num_mh_steps: int = 1, mcmc_step_size: float = 1e-3, noise_std=0.01):
        key_enc, key_vf = jr.split(key)
        self.param_dim = param_dim
        self.latent_dim = latent_dim

        self.encoder = TimeSeriesEncoder(feature_dim, hidden_size_rnn, latent_dim, key_enc)
        self.vector_field = VectorField(param_dim, latent_dim, hidden_sizes[0], depth=len(hidden_sizes), key=key_vf)
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

    def sample_with_MCMC(
        self, cond: jnp.ndarray, key: jax.Array, num_samples: int, steps: int = 100
    ) -> jnp.ndarray:
        """Generate samples from the learned distribution."""
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = batch_size * num_samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)

        z0 = jr.normal(key, (total_num, self.param_dim))

        # Simple Euler Integration (or RK4)
        # Since we use OT path, the trajectory is straight, so Euler is often sufficient.
        dt = 1.0 / steps

        def body_fn(i, z_curr):
            t = i * dt
            v = jax.vmap(self.vector_field, in_axes=(None, 0, 0))(t, z_curr, cond_emb_rep)
            return z_curr + v * dt

        z1 = jax.lax.fori_loop(0, steps, body_fn, z0)
        if self.refinement:
            s_key = jr.split(key, 2)[1]
            z1 = self.refinement(z1, cond_rep, s_key, cond_emb_rep)[0]
        # Reshape to (num_samples, batch_size, dim)
        samples = z1.reshape(batch_size, num_samples, -1)
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
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state, optimizer, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return cfm_loss(full_model, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
    else:
        loss_fn = lambda m: cfm_loss(m, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
    return model, opt_state, loss