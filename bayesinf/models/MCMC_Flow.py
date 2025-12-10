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
from typing import Any, Optional, List, Tuple, Callable
from networks import TimeSeriesEncoder
from normalizing_flow import ActNorm, RandomPermutation, AffineCoupling, SplineCoupling

class MALATransition(eqx.Module):
    log_step_size: jnp.ndarray
    beta: float
    
    def __init__(self, dim: int, beta: float, initial_step_size: float = 1e-3):
        # Learnable step size
        self.log_step_size = jnp.full((dim,), jnp.log(initial_step_size))
        self.beta = beta

    @property
    def step_size(self):
        return jnp.exp(self.log_step_size)

    def __call__(
        self, 
        x: jnp.ndarray, 
        target_energy_fn: Callable[[jnp.ndarray], jnp.ndarray], 
        key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Performs one MALA step.
        x: (Batch, Dim)
        target_energy_fn: Function u -> Energy (Scalar). Target ~ exp(-Energy)
        """
        k1, k2 = jr.split(key)
        eps = self.step_size # (Dim,) broadcastable to (Batch, Dim)
        D = x.shape[-1]

        # 1. Calculate Energy & Gradient at Current Position
        # We use the gradient of the SUM of energies. 
        # Since samples in a batch are independent, grad_x[i] depends only on energy[i].
        def sum_energy(val):
            return jnp.sum(target_energy_fn(val))
            
        E_x, grad_E_x = jax.value_and_grad(sum_energy)(x)
        grad_E_x = jax.lax.stop_gradient(grad_E_x) 

        # Note: E_x here is the scalar sum. We need the individual energies for the MH ratio.
        # However, value_and_grad gives the sum. Let's get individual energies separately 
        # or rely on the fact we need to evaluate it anyway.
        # To be efficient, let's just evaluate the function to get the vector.
        E_x_vec = target_energy_fn(x) 
        
        grad_log_p_x = -grad_E_x # (Batch, Dim)
        
        # 2. Proposal
        noise = jr.normal(k1, x.shape)
        mean_x = x + eps * grad_log_p_x
        x_prop = mean_x + jnp.sqrt(2 * eps) * noise
        
        # 3. Energy & Gradient at Proposal
        # Again, compute gradients via sum trick
        grad_E_prop = jax.grad(sum_energy)(x_prop)
        grad_E_prop = jax.lax.stop_gradient(grad_E_prop)
        E_prop_vec = target_energy_fn(x_prop)
        
        grad_log_p_prop = -grad_E_prop
        mean_prop = x_prop + eps * grad_log_p_prop

        # 4. MH Ratio (in log space) - All operations elementwise on (Batch,)
        # sum over Dim axis=-1
        log_q_fwd = -0.5 * jnp.sum((x_prop - mean_x)**2 / (2 * eps), axis=-1)  - (D/2) * jnp.sum(jnp.log(2 * jnp.pi * eps)) # Simplified constant
        log_q_bwd = -0.5 * jnp.sum((x - mean_prop)**2 / (2 * eps), axis=-1) - (D/2) * jnp.sum(jnp.log(2 * jnp.pi * eps))
        
        # log_alpha = log_pi(x') - log_pi(x) + log_q(x|x') - log_q(x'|x)
        # log_pi = -E
        log_alpha = (-E_prop_vec + E_x_vec) + (log_q_bwd - log_q_fwd)
        
        # 5. Accept/Reject
        accept_prob = jnp.exp(jnp.minimum(0.0, log_alpha)) # (Batch,)
        
        # Sample uniform for each batch item
        u = jr.uniform(k2, accept_prob.shape)
        accepted = u < accept_prob
        
        # Stop gradient on acceptance mask
        accepted = jax.lax.stop_gradient(accepted)
        
        # Broadcast acceptance mask to (Batch, Dim)
        accepted_mask = accepted[:, None]
        
        new_x = jnp.where(accepted_mask, x_prop, x)
        
        return new_x, accept_prob


class StochasticCondNormFlow(eqx.Module):
    dim: int = eqx.field(static=True)
    encoder: TimeSeriesEncoder
    layers: List[eqx.Module]

    def __init__(
        self, 
        cond_dim: int,         
        param_dim: int, 
        *,
        key: jax.Array,
        n_coupling: int = 4, 
        hidden_sizes_nf: int = 64, 
        num_hidden_nf: int = 2,
        encoded_latent_dim: int = 16,
        hidden_rnn: int = 64,
        flow_type = 'realnvp',
        rnn_type: str = 'lstm',
        mcmc_step_size: float = 1e-3
    ):
        self.dim = param_dim
        k_enc, k_flow = jax.random.split(key)
        self.encoder = TimeSeriesEncoder(cond_dim, hidden_rnn, encoded_latent_dim, k_enc, rnn=rnn_type)
        
        self.layers = []
        keys = jax.random.split(k_flow, n_coupling + 1)
        
        def get_mask(parity):
            return jnp.array([(i + parity) % 2 for i in range(param_dim)], dtype=jnp.float32)

        flow_type = flow_type.lower()
        
        def get_mask(parity):
            return jnp.array(
                [(i + parity) % 2 for i in range(param_dim)], 
                dtype=jnp.float32
            )
        
        for i in range(n_coupling // 2):
            k_layer1, k_layer2, k_perm = jax.random.split(keys[i], 3)
            
            # ActNorm for stability
            self.layers.append(ActNorm(param_dim))
            
            # Random permutation
            self.layers.append(RandomPermutation(param_dim, key=k_perm))
            
            # Coupling layers based on flow type
            if flow_type == 'nsf':
                self.layers.append(
                    SplineCoupling(
                        param_dim, 
                        [hidden_sizes_nf] * num_hidden_nf,
                        encoded_latent_dim,
                        get_mask(i % 2),
                        key=k_layer1,
                        num_bins=4
                    )
                )
                self.layers.append(
                    SplineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        encoded_latent_dim,
                        get_mask((i + 1) % 2),
                        key=k_layer2,
                        num_bins=4
                    )
                )
            else:  # RealNVP (default)
                self.layers.append(
                    AffineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        encoded_latent_dim,
                        get_mask(i % 2),
                        key=k_layer1
                    )
                )
                self.layers.append(
                    AffineCoupling(
                        param_dim,
                        [hidden_sizes_nf] * num_hidden_nf,
                        encoded_latent_dim,
                        get_mask((i + 1) % 2),
                        key=k_layer2
                    )
                )
            if i == n_coupling // 4 -1:
                beta_fwd = (i + 1) / n_coupling
                self.layers.append(MALATransition(param_dim, beta=beta_fwd, initial_step_size=mcmc_step_size))
        
        # Final ActNorm
        self.layers.append(ActNorm(param_dim))

    def _compute_interpolated_energy(
        self, 
        u_batch: jnp.ndarray, 
        cond_emb_batch: jnp.ndarray, 
        state_obs: jnp.ndarray, deriv_obs: Optional[jnp.ndarray], 
        dynamics_fn: Callable,
        dt: float, t1: float,
        idx_current: int,
        beta: float
    ):
        """
        Computes E(u) for a BATCH of particles.
        u_batch: (Batch, Dim)
        Returns: (Batch,)
        """
        
        # --- A. Energy from Z (Prior) ---
        # Project u -> z using remaining Forward layers
        z_curr = u_batch
        ldj_fwd = jnp.zeros(u_batch.shape[0])
        
        # Layers after current MALA layer
        # Note: self.layers is fixed. idx_current is the index of the MALA layer in that list.
        forward_layers = self.layers[idx_current+1:]
        
        for layer in forward_layers:
            if isinstance(layer, MALATransition): continue
            
            if isinstance(layer, (ActNorm, RandomPermutation)):
                z_curr, ldj = layer(z_curr, reverse=False)
            else:
                z_curr, ldj = layer(z_curr, cond_emb_batch, reverse=False)
            
            # ActNorm/Coupling return (Batch,) or (Batch, ...) ldj, verify shape
            if ldj.ndim > 1: ldj = jnp.sum(ldj, axis=-1)
            ldj_fwd += ldj
            
        # E_z = -log p(z) - log |det J|
        # p(z) is Standard Normal. Sum over Dim axis.
        E_prior_z = 0.5 * jnp.sum(z_curr**2, axis=-1) # + const
        E_z_branch = E_prior_z - ldj_fwd

        # --- B. Energy from Theta (Posterior) ---
        # Project u -> theta using previous layers in Inverse
        theta_curr = u_batch
        ldj_inv = jnp.zeros(u_batch.shape[0])
        
        # Layers before current MALA layer (reversed)
        backward_layers = list(reversed(self.layers[:idx_current]))
        
        for layer in backward_layers:
            if isinstance(layer, MALATransition): continue
            
            if isinstance(layer, (ActNorm, RandomPermutation)):
                theta_curr, ldj = layer(theta_curr, reverse=True)
            else:
                theta_curr, ldj = layer(theta_curr, cond_emb_batch, reverse=True)
            
            if ldj.ndim > 1: ldj = jnp.sum(ldj, axis=-1)
            ldj_inv += ldj

        # E_theta = -log p(theta) - log p(D|theta) - log |det J_inv|
        # 1. Prior on Theta
        E_prior_theta = 0.5 * jnp.sum(theta_curr**2, axis=-1)
        
        # 2. Likelihood (Equation Error)
        E_likelihood = jnp.zeros(u_batch.shape[0])
        
        if dynamics_fn is not None and state_obs is not None:
            # cond_raw_batch: (Batch, T, StateDim)
            # theta_curr: (Batch, Dim)
            
            # Finite Difference Derivatives
            # shape (Batch, T-1, StateDim)
            
            theta_phys = jax.nn.softplus(theta_curr) 
            
            # Helper to map dynamics over Time and State
            # dynamics_fn: (t, state, theta) -> deriv
            ts = jnp.linspace(0, t1-dt, state_obs.shape[1])

            if deriv_obs is not None:
                # We use vmap to handle Batch dimension, and inner vmap for Time dimension
                # outer vmap: batch (b)
                # inner vmap: time (t)
                def compute_traj_deriv(states, theta):
                    # states: (T, S), theta: (D)
                    return jax.vmap(lambda _t, _s: dynamics_fn(_t, _s, theta))(ts, states)

                # (Batch, T, S)
                pred_deriv = jax.vmap(compute_traj_deriv)(state_obs, theta_phys)
                
                residuals = deriv_obs - pred_deriv
            else:
                # dynamics_fn is integral of ode
                y0_batch = jnp.stack([state_obs[:, 0, 0]/2, jnp.zeros(state_obs.shape[0]), 
                                    state_obs[:, 0, 0]/2, jnp.zeros(state_obs.shape[0])], axis=-1)
                # Batch solve
                pred_state = jax.vmap(dynamics_fn, in_axes=(0, 0, None))(y0_batch, theta_phys, ts)

                residuals = state_obs - pred_state

            
            # Sum over Time and State dimensions, keep Batch
            mse = jnp.sum(residuals**2, axis=(1, 2))
            
            sigma = 0.1 
            E_likelihood = 0.5 * mse / (sigma**2)

        E_theta_branch = E_prior_theta + E_likelihood - ldj_inv
        
        # --- C. Interpolate ---
        return beta * E_z_branch + (1.0 - beta) * E_theta_branch

    def _flow_call(self, 
                   x: jnp.ndarray, 
                   state_obs: jnp.ndarray, 
                   deriv_obs: Optional[jnp.ndarray], 
                   cond_emb: jnp.ndarray,
                   dynamics_fn: Callable,
                   dt: float, t1: float,
                   key: jax.Array,
                   reverse: False
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        logdet = jnp.zeros(x.shape[:-1])
        z = x
        total_layers = len(self.layers)
        
        # Determine the sequence to iterate
        if reverse:
            # Create a list of (layer, absolute_index)
            layers_with_idx = [(self.layers[i], i) for i in reversed(range(total_layers))]
        else:
            layers_with_idx = [(self.layers[i], i) for i in range(total_layers)]
            
        for i, (layer, abs_idx) in enumerate(layers_with_idx):
            if isinstance(layer, MALATransition):
                beta = layer.beta 
                k_step = jr.fold_in(key, i) # Fold in loop index for randomness
                
                # Closure captures the ABSOLUTE index `abs_idx`
                def energy_closure(u_batch):
                    return self._compute_interpolated_energy(
                        u_batch, cond_emb, state_obs, deriv_obs, dynamics_fn, dt, t1, abs_idx, beta
                    )
                
                # Pass the BATCH directly to the MALA layer
                # The MALA layer handles gradients internally via sum-grad trick
                z, _ = layer(z, energy_closure, k_step)
                
            elif isinstance(layer, (ActNorm, RandomPermutation)):
                z, ldj = layer(z, reverse=reverse)
                logdet += ldj
            else:
                z, ldj = layer(z, cond_emb, reverse=reverse)
                logdet += ldj

        return z, logdet


    def __call__(self, 
                 x: jnp.ndarray, 
                 cond: jnp.ndarray, 
                 dynamics_fn: Callable,
                 dt: float, t1: float,
                 key: jax.Array, 
                 multi_modal=False
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward Pass (Training): x -> z
        Includes MALA transitions to train the flow with the 'Intertwined' objective.
        """
        cond_emb = self.encoder(cond)
        state_obs = cond[:, :-1, :]
        deriv_obs = (cond[:, 1:, :] - cond[:, :-1, :]) / dt if not multi_modal else None
        z, logdet = self._flow_call(x, state_obs, deriv_obs, cond_emb, dynamics_fn, dt, t1, key, reverse=False)
        return z, logdet

    def log_prob(self, 
                 x: jnp.ndarray, 
                 cond: jnp.ndarray, 
                 dynamics_fn: Callable,
                 dt: float, t1: float,
                 key: jax.Array,
                 multi_modal=False) -> jnp.ndarray:
        z, logdet = self(x, cond, dynamics_fn, dt, t1, key, multi_modal=multi_modal)
        # Prior log prob (Gaussian)
        # -0.5 * (z^T z + D * log(2pi))
        log_prob_z = -0.5 * (jnp.sum(z**2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return log_prob_z + logdet

    def inverse(self, 
                z: jnp.ndarray, 
                cond: jnp.ndarray, 
                dynamics_fn: Callable,
                dt: float, t1: float,
                key: jax.Array,
                multi_modal=False) -> jnp.ndarray:
        """
        Inverse Pass (Training): z -> x
        Includes MALA transitions to train the flow with the 'Intertwined' objective.
        """
        cond_emb = self.encoder(cond)
        state_obs = cond[:, :-1, :]
        deriv_obs = (cond[:, 1:, :] - cond[:, :-1, :]) / dt if not multi_modal else None
        x, logdet = self._flow_call(z, state_obs, deriv_obs, cond_emb, dynamics_fn, dt, t1, key, reverse=True)
        return x, logdet       

    def sample(
        self, 
        cond: jnp.ndarray, 
        num_samples: int,
        dynamics_fn: Callable, 
        dt:float, t1:float,
        key: jax.Array,
        multi_modal=False
    ) -> jnp.ndarray:
        """
        Inverse Pass (Sampling): Z -> Theta
        """
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = batch_size * num_samples
        z = jr.normal(key, (total_num, self.dim))
        state_obs = cond[:, :-1, :]
        deriv_obs = (cond[:, 1:, :] - cond[:, :-1, :]) / dt if not multi_modal else None
        # Expand condition
        state_obs_flat = jnp.repeat(state_obs, num_samples, axis=0) # (B*K, T, C)
        deriv_obs_flat = jnp.repeat(deriv_obs, num_samples, axis=0) if deriv_obs is not None else None
        cond_emb_flat = jnp.repeat(cond_emb, num_samples, axis=0)
        
        samples = self._flow_call(z, state_obs_flat, deriv_obs_flat, cond_emb_flat, dynamics_fn, dt, t1, key, reverse=True)[0]
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples


def snf_loss(model, true_params: jnp.ndarray, cond:jnp.ndarray, dynamics_fn, dt, t1, key, multi_modal=False):
    ml_loss = model.log_prob(true_params, cond, dynamics_fn, dt, t1, key, multi_modal=multi_modal)
    loss = - ml_loss.mean()
    return loss

@eqx.filter_jit
def make_step(model, x: jnp.ndarray, true_params: jnp.ndarray, 
              dynamics_fn: Callable, dt, t1, key, opt_state: Any, 
              optimizer: optax.GradientTransformation, multi_modal=False):

    loss_fn = lambda m: snf_loss(m, true_params, x, dynamics_fn, dt, t1, key, multi_modal=multi_modal)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss