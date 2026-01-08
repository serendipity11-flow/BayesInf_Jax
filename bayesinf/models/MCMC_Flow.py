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
from typing import Any, Callable, List, Tuple
from bayesinf.models.networks import TimeSeriesEncoder
from bayesinf.models.normalizing_flow import ActNorm, RandomPermutation

class AffineCoupling(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    idx_a: jnp.ndarray
    idx_b: jnp.ndarray
    net: eqx.nn.MLP

    def __init__(
        self, 
        input_dim: int, 
        hidden_sizes: List[int], 
        cond_dim: int, 
        mask: jnp.ndarray, 
        key: jax.Array
    ):
        self.dim = input_dim
        self.cond_dim = cond_dim

        self.idx_a = jnp.where(mask)[0]  # Indices where mask=1 (conditioner input)
        self.idx_b = jnp.where(1 - mask)[0] # Indices where mask=0 (transformed)

        in_dim = int(jnp.sum(mask)) + cond_dim
        out_dim = input_dim - int(jnp.sum(mask))
        
        # Equinox MLP
        # We want the final layer initialized to near-zero.
        # Equinox doesn't allow easy custom init per layer in the constructor
        # simply, so we construct, then modify.
        self.net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim * 2,
            width_size=hidden_sizes[0] if hidden_sizes else 16, # simplify for list
            depth=len(hidden_sizes) + 1, # depth includes output layer
            activation=jax.nn.relu,
            key=key
        )
        
        # Initialize last layer to zeros for stability (identity init)
        final_layer_idx = len(self.net.layers) - 1
        w_zeros = jnp.zeros_like(self.net.layers[final_layer_idx].weight)
        b_zeros = jnp.zeros_like(self.net.layers[final_layer_idx].bias)
        
        new_final = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.net.layers[final_layer_idx],
            (w_zeros, b_zeros)
        )
        self.net = eqx.tree_at(
            lambda n: n.layers[final_layer_idx],
            self.net,
            new_final
        )

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, reverse: bool = False):
        # x: (batch, dim), cond: (batch, cond_dim)

        # Function to process a single batch element
        def _step(x_i, c_i):
            x_a_flat = x_i[..., self.idx_a]
            inp = jnp.concatenate([x_a_flat, c_i], axis=-1)
            st = self.net(inp)
            s, t = jnp.split(st, 2, axis=-1)
            s = jnp.tanh(s)
            
            x_b_flat = x_i[..., self.idx_b]
            
            if not reverse:
                z_b_flat = x_b_flat * jnp.exp(s) + t
                # Reconstruct full z
                z = jnp.zeros_like(x_i)
                z = z.at[..., self.idx_a].set(x_a_flat)
                z = z.at[..., self.idx_b].set(z_b_flat)
                ldj = jnp.sum(s)
                return z, ldj
            else:
                x_b_rec = (x_b_flat - t) * jnp.exp(-s)
                z = jnp.zeros_like(x_i)
                z = z.at[..., self.idx_a].set(x_a_flat)
                z = z.at[..., self.idx_b].set(x_b_rec)
                ldj = -jnp.sum(s)
                return z, ldj

        return jax.vmap(_step)(x, cond)

class MALALayer(eqx.Module):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) layer for Bayesian inference.
    
    This layer performs MCMC sampling using Langevin dynamics as a proposal.
    Uses stop_gradient to prevent backpropagation through the MCMC steps.
    
    Args:
        log_posterior: Function that computes log posterior given (x, condition)
        lambd: Interpolation parameter between prior (0) and posterior (1)
        num_mh_steps: Number of Metropolis-Hastings steps
        num_langevin_steps: Number of Langevin steps for proposal
        step_size: Step size for Langevin dynamics
    """
    
    log_posterior: Callable
    lambd: float
    num_mh_steps: int
    num_langevin_steps: int
    step_size: float
    
    def __init__(
        self,
        log_posterior: Callable,
        lambd: float = 1.0,
        num_mh_steps: int = 3,
        num_langevin_steps: int = 1,
        step_size: float = 5e-3
    ):
        self.log_posterior = log_posterior
        self.lambd = lambd
        self.num_mh_steps = num_mh_steps
        self.num_langevin_steps = num_langevin_steps
        self.step_size = step_size
    
    def get_interpolated_energy(
        self, x, condition):
        """
        Compute interpolated energy between prior and posterior.
        
        Energy = lambd * log_posterior(x|y) + (1-lambd) * 0.5 * ||x||^2
        """
        if self.lambd == 1.0:
            return 0.5 * jnp.sum(x**2, axis=-1)
        elif self.lambd == 0.0:
            return self.log_posterior(x, condition)
        else:
            log_post = self.log_posterior(x, condition)
            prior_energy = 0.5 * jnp.sum(x**2, axis=-1)
            return self.lambd * log_post + (1.0 - self.lambd) * prior_energy
    
    def energy_grad(
        self, x, condition):
        """Compute gradient of energy and energy value."""
        def energy_fn(x_single):
            return self.get_interpolated_energy(
                x_single[None, :], condition[0:1, :]
            )[0]
        
        # Vectorized gradient computation
        grad_fn = jax.vmap(jax.grad(energy_fn))
        energy_fn_vec = lambda x: self.get_interpolated_energy(x, condition)
        
        grad_energy = grad_fn(x)
        energy_val = energy_fn_vec(x)
        
        return grad_energy, energy_val
    
    def langevin_proposal(
        self, x, condition, key):
        """
        Generate proposal using Langevin dynamics.
        
        Returns:
            x_prop: Proposed samples
            log_det: Log determinant of proposal transition
            energy_x: Energy at current x
            energy_prop: Energy at proposed x
        """
        beta = 1.0
        step_size = self.step_size / max(self.lambd, 1e-8)
        
        log_det = jnp.zeros(x.shape[0])
        x_curr = x
        
        # Initial energy
        grad_x, energy_x_init = self.energy_grad(x_curr, condition)
        
        # Langevin steps
        for i in range(self.num_langevin_steps):
            key, subkey = jax.random.split(key)
            
            # Compute gradient at current position
            grad_x, energy_x = self.energy_grad(x_curr, condition)
            
            # Langevin proposal step
            eta = jax.random.normal(subkey, x_curr.shape)
            noise_scale = jnp.sqrt(2 * step_size / beta)
            x_prop = x_curr - step_size * grad_x + noise_scale * eta
            
            # Compute gradient at proposed position
            grad_prop, energy_prop = self.energy_grad(x_prop, condition)
            
            # Reverse transition kernel
            eta_reverse = (x_curr - x_prop + step_size * grad_prop) / noise_scale
            
            # Update log determinant (for detailed balance)
            log_det += 0.5 * jnp.sum(eta**2 - eta_reverse**2, axis=-1)
            
            x_curr = x_prop
        
        return x_prop, log_det, energy_x_init, energy_prop
    
    def mh_step(
        self,
        x_curr,
        condition,
        key
    ):
        """
        Single Metropolis-Hastings step with Langevin proposal.
        
        Returns:
            x_new: Accepted or rejected samples
            energy_diff: Energy difference (for tracking)
        """
        key_prop, key_accept = jax.random.split(key)
        
        # Generate proposal using Langevin dynamics
        x_prop, log_det, energy_curr, energy_prop = self.langevin_proposal(
            x_curr, condition, key_prop
        )
        
        # Metropolis-Hastings acceptance ratio
        # log(alpha) = -energy_prop + energy_curr + log_det
        log_alpha = -energy_prop + energy_curr + log_det
        
        # Accept/reject
        u = jax.random.uniform(key_accept, shape=(x_curr.shape[0],))
        accept = (jnp.log(u) < log_alpha).astype(jnp.float32)
        
        # Update samples
        accept_expanded = accept[:, None]
        x_new = accept_expanded * x_prop + (1.0 - accept_expanded) * x_curr
        
        # Track energy difference
        energy_diff = accept * energy_prop + (1.0 - accept) * energy_curr
        
        return x_new, energy_diff
    
    def __call__(
        self,
        x,
        condition,
        key
    ):
        """
        Forward pass: perform MALA sampling with stop_gradient.
        
        Args:
            x: Input samples
            condition: Conditioning information
            key: PRNG key for randomness
            
        Returns:
            x_out: Output samples after MALA steps
            log_det: Log determinant term (energy change)
        """
        # Initial energy
        energy_init = self.get_interpolated_energy(x, condition)
        
        x_curr = x
        
        # Perform multiple MH steps
        for i in range(self.num_mh_steps):
            key, subkey = jax.random.split(key)
            x_curr, energy_final = self.mh_step(x_curr, condition, subkey)
        
        # x_out = jax.lax.stop_gradient(x_curr)
        x_out = x_curr
        
        # Compute log determinant (energy difference)
        energy_final = self.get_interpolated_energy(x_out, condition)
        log_det = energy_final - energy_init
        
        # Stop gradient on log_det as well
        log_det = jax.lax.stop_gradient(log_det)
        
        return x_out, log_det

class CondSNF(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    encoder: TimeSeriesEncoder
    transforms: List[eqx.Module]
    log_posterior_fn: Callable = eqx.field(static=True)

    def __init__(
        self, 
        dim: int, 
        cond_dim: int, 
        log_posterior_fn: Callable,
        key: jax.Array,
        n_coupling: int = 6, 
        hidden_nf: int = 128, 
        num_hidden_nf: int = 2,
        encoded_latent_dim: int = 16, 
        hidden_rnn: int = 128,
        rnn_type: str = "lstm",
        flow_type: str = 'realnvp',
        metr_steps_per_block: int = 3,
        lang_steps: int = 1,
        stepsize: float = 5e-3
    ):
        self.dim = dim
        self.cond_dim = cond_dim
        self.log_posterior_fn = log_posterior_fn
        
        k_enc, k_flow = jax.random.split(key)
        self.encoder = TimeSeriesEncoder(cond_dim, hidden_rnn, encoded_latent_dim, k_enc, rnn=rnn_type)
        
        self.transforms = []
        keys = jax.random.split(k_flow, n_coupling + 2) # +2 for ActNorms

        flow_type = flow_type.lower()
        
        def get_mask(parity):
            return jnp.array([(i + parity) % 2 for i in range(dim)], dtype=jnp.float32)

        for i in range(n_coupling // 2):
            k_layer1, k_layer2, k_perm = jax.random.split(keys[i], 3)
            
            # ActNorm
            self.transforms.append(ActNorm(dim))
            
            # Permutation
            self.transforms.append(RandomPermutation(dim, key=k_perm))
            
            # Coupling / Flow Step
            # RealNVP
            self.transforms.append(
                AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                get_mask(i % 2), key=k_layer1)
            )
            self.transforms.append(
                AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                get_mask((i + 1) % 2), key=k_layer2)
            )
            lambd = (i + 1) / (n_coupling // 2)
            if lambd == 1.0:
                self.transforms.append(
                    MALALayer(log_posterior_fn, lambd, metr_steps_per_block, lang_steps, stepsize)
                )

        # Final ActNorm
        self.transforms.append(ActNorm(dim))

    def __flow_call__(self, z: jnp.ndarray, cond:jnp.ndarray, cond_emb:jnp.ndarray, key: jax.Array, reverse=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(z.shape[:-1])
        x = z
        num_stochastic_layers = len([t for t in self.transforms if isinstance(t, MALALayer)])
        keys = jr.split(key, num_stochastic_layers)
        key_id = num_stochastic_layers - 1 if reverse else 0

        flow_list = self.transforms if not reverse else reversed(self.transforms)
        for t in flow_list:
            if isinstance(t, (ActNorm, RandomPermutation)):
                x, ldj = t(x, reverse=reverse)
            elif isinstance(t, MALALayer):
                x, ldj = t(x, cond, keys[key_id])
                key_id += -1 if reverse else 1
            else:
                x, ldj = t(x, cond_emb, reverse=reverse)
            logdet = logdet + ldj

        return x, logdet

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # cond: raw time series (Batch, T, C)
        # Encode condition
        cond_emb = self.encoder(cond)
        z, logdet = self.__flow_call__(x, cond, cond_emb, key, reverse=False)
        return z, logdet   

    def inverse(self, z: jnp.ndarray, cond: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cond_emb = self.encoder(cond)
        x, logdet = self.__flow_call__(z, cond, cond_emb, key, reverse=True)
        return x, logdet

    def log_prob(self, x: jnp.ndarray, cond: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        z, logdet = self(x, cond, key)
        # Prior log prob (Gaussian)
        # -0.5 * (z^T z + D * log(2pi))
        log_prob_z = -0.5 * (jnp.sum(z**2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return log_prob_z + logdet

    def sample(self, cond: jnp.ndarray, key: jax.Array, num_samples: int) -> jnp.ndarray:
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)
        z_key, s_key = jr.split(key, 2)
        z = jr.normal(z_key, (total_num, self.dim))
        samples = self.__flow_call__(z, cond_rep, cond_emb_rep, s_key, reverse=True)[0]
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples


def snf_loss(model, y: jnp.ndarray, true_params: jnp.ndarray, key:jax.Array):
    ml_loss = model.log_prob(true_params, y, key)
    loss = - ml_loss.mean()
    return loss


@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state: Any, optimizer: optax.GradientTransformation):
    loss_fn = lambda m: snf_loss(m, y, true_params, key)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss