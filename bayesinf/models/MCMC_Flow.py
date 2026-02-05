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
from typing import Callable, Tuple, List, Optional
from bayesinf.models.networks import TimeSeriesEncoder

class ActNorm(eqx.Module):
    bias: jnp.ndarray
    logs: jnp.ndarray
    logscale_factor: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    initialized: bool

    def __init__(self, dim: int, logscale_factor=1.0, eps=1e-6):
        self.bias = jnp.zeros((dim,))
        self.logs = jnp.zeros((dim,))
        self.logscale_factor = logscale_factor
        self.eps = eps
        self.initialized = False

    def initialize(self, x: jnp.ndarray):
        """
        Data-dependent initialization.
        Returns a NEW ActNorm instance with updated parameters.
        """
        # x: (batch, dim)
        mean = jnp.mean(x, axis=0)
        std = jnp.std(x, axis=0) + self.eps
        
        new_bias = -mean
        new_logs = jnp.log(1.0 / (std + self.eps)) / self.logscale_factor
        
        # Return a copy of self with updated fields
        return eqx.tree_at(
            lambda t: (t.bias, t.logs, t.initialized), 
            self, 
            (new_bias, new_logs, True)
        )

    def __call__(self, x: jnp.ndarray, reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, dim)
        # Note: In JAX/Equinox, we cannot check "if not self.initialized: self.initialize(x)"
        # inside the forward pass easily if JIT compiling. 
        # The user should call .initialize() explicitly before training.
        
        logs = self.logs * self.logscale_factor
        
        if not reverse:
            z = (x + self.bias) * jnp.exp(logs)
            # ldj = sum(logs) for each batch element
            ldj_val = jnp.sum(logs)
            ldj = jnp.full(x.shape[:-1], ldj_val)
            return z, ldj
        else:
            z = x * jnp.exp(-logs) - self.bias
            ldj_val = -jnp.sum(logs)
            ldj = jnp.full(x.shape[:-1], ldj_val)
            return z, ldj


class RandomPermutation(eqx.Module):
    dim: int = eqx.field(static=True)
    perm: jnp.ndarray
    inv_perm: jnp.ndarray

    def __init__(self, dim: int, key: jax.Array):
        self.dim = dim
        self.perm = jax.random.permutation(key, jnp.arange(dim))
        self.inv_perm = jnp.argsort(self.perm)

    def __call__(self, x: jnp.ndarray, reverse: bool = False):
        if not reverse:
            return x[..., self.perm], jnp.zeros(x.shape[:-1])
        else:
            return x[..., self.inv_perm], jnp.zeros(x.shape[:-1])

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
    apply_gradient_schedule: bool
    def __init__(
        self,
        log_posterior: Callable,
        lambd: float = 1.0,
        num_mh_steps: int = 1,
        num_langevin_steps: int = 1,
        step_size: float = 1e-3,
        apply_gradient_schedule=False
    ):
        self.log_posterior = log_posterior
        self.lambd = lambd
        self.num_mh_steps = num_mh_steps
        self.num_langevin_steps = num_langevin_steps
        self.step_size = step_size
        self.apply_gradient_schedule = apply_gradient_schedule
    
    def get_interpolated_energy(
        self, x, condition, gradient_scale):
        """
        Compute interpolated energy between prior and posterior.
        
        Energy = lambd * log_posterior(x|y) + (1-lambd) * 0.5 * ||x||^2
        """
        if self.lambd < 0.001:
            return 0.5 * jnp.sum(x**2, axis=-1)
        elif self.lambd > 0.999:
            return self.log_posterior(x, condition, gradient_scale)[0]
        else:
            log_post = self.log_posterior(x, condition, gradient_scale)[0]
            prior_energy = 0.5 * jnp.sum(x**2, axis=-1)
            return self.lambd * log_post + (1.0 - self.lambd) * prior_energy
    
    def energy_grad(
        self, x, condition, gradient_scale):
        """Compute gradient of energy and energy value."""
        def energy_fn(x_single):
            return self.get_interpolated_energy(
                x_single[None, :], condition[0:1, :], gradient_scale
            )[0]
        
        # Vectorized gradient computation
        grad_fn = jax.vmap(jax.grad(energy_fn))
        energy_fn_vec = lambda x: self.get_interpolated_energy(x, condition, gradient_scale)
        
        grad_energy = jax.lax.stop_gradient(grad_fn(x))
        energy_val = energy_fn_vec(x)
        
        return grad_energy, energy_val
    
    def langevin_proposal(
        self, x, condition, key, gradient_scale):
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
        grad_x, energy_x_init = self.energy_grad(x_curr, condition, gradient_scale)
        
        # Langevin steps
        for i in range(self.num_langevin_steps):
            key, subkey = jax.random.split(key)
            
            # Compute gradient at current position
            grad_x, energy_x = self.energy_grad(x_curr, condition, gradient_scale)
            
            # Langevin proposal step
            eta = jax.random.normal(subkey, x_curr.shape)
            noise_scale = jnp.sqrt(2 * step_size / beta)
            x_prop = x_curr - step_size * grad_x + noise_scale * eta
            
            # Compute gradient at proposed position
            grad_prop, energy_prop = self.energy_grad(x_prop, condition, gradient_scale)
            
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
        key,
        gradient_scale
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
            x_curr, condition, key_prop, gradient_scale
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
        key,
        cond_emb=None,
        gradient_scale=0
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
        energy_init = self.get_interpolated_energy(x, condition, gradient_scale)
        
        # x_curr = jax.lax.stop_gradient(x)
        x_curr = x 
        
        # Perform multiple MH steps
        for i in range(self.num_mh_steps):
            key, subkey = jax.random.split(key)
            x_curr, energy_final = self.mh_step(x_curr, condition, subkey, gradient_scale)

        x_out = x_curr

        # Compute log determinant (energy difference)
        energy_final = self.get_interpolated_energy(x_out, condition, gradient_scale)
        log_det = energy_final - energy_init
        
        # Stop gradient on log_det as well
        if self.apply_gradient_schedule:
            log_det = gradient_schedule(log_det, gradient_scale)
        else:
            log_det = jax.lax.stop_gradient(log_det)
        
        return x_out, log_det


class MCMCLayer(eqx.Module):
    """Simple MCMC layer with gradient scheduling."""
    
    log_posterior: Callable
    lambd: float
    num_mh_steps: int
    noise_std: float
    apply_gradient_schedule: bool
    
    def __init__(
        self,
        log_posterior: Callable,
        lambd: float = 1.0,
        num_mh_steps: int = 3,
        noise_std: float = 0.1,
        apply_gradient_schedule: bool = False
    ):
        self.log_posterior = log_posterior
        self.lambd = lambd
        self.num_mh_steps = num_mh_steps
        self.noise_std = noise_std
        self.apply_gradient_schedule = apply_gradient_schedule
    
    def get_interpolated_energy(self, x, condition, gradient_scale):
        
        if self.lambd >= 0.999:
            return self.log_posterior(x, condition, gradient_scale)[0]
        elif self.lambd <= 0.001:
            return 0.5 * jnp.sum(x**2, axis=-1)
        else:
            log_post = self.log_posterior(x, condition, gradient_scale)[0]
            prior_energy = 0.5 * jnp.sum(x**2, axis=-1)
            return self.lambd * log_post + (1.0 - self.lambd) * prior_energy
    
    def mh_step(self, x_curr, condition, key, gradient_scale):
        key_prop, key_accept = jax.random.split(key)
        
        energy_curr = self.get_interpolated_energy(x_curr, condition, gradient_scale)
        
        noise = self.noise_std * jax.random.normal(key_prop, x_curr.shape)
        x_prop = x_curr + noise
        
        energy_prop = self.get_interpolated_energy(x_prop, condition, gradient_scale)
        
        log_alpha = -energy_prop + energy_curr
        log_alpha = jnp.clip(log_alpha, -100.0, 10.0)
        
        u = jax.random.uniform(key_accept, shape=(x_curr.shape[0],))
        accept = (jnp.log(u + 1e-10) < log_alpha).astype(jnp.float32)
        
        accept_expanded = accept[:, None]
        x_new = accept_expanded * x_prop + (1.0 - accept_expanded) * x_curr
        
        return x_new, accept
    
    def __call__(
        self,
        x,
        condition,
        key,
        cond_emb=None,
        gradient_scale: float = 1.0
    ):
        energy_init = self.get_interpolated_energy(x, condition, gradient_scale)
        
        # x_curr = jax.lax.stop_gradient(x) 
        x_curr = x
        for i in range(self.num_mh_steps):
            key, subkey = jax.random.split(key)
            x_curr, accept = self.mh_step(x_curr, condition, subkey, gradient_scale)
        x_out = x_curr
        
        energy_final = self.get_interpolated_energy(x_out, condition, gradient_scale)
        log_det = energy_final - energy_init
        
        if self.apply_gradient_schedule:
            log_det = gradient_schedule(log_det, gradient_scale)
        else:
            log_det = jax.lax.stop_gradient(log_det)
        
        return x_out, log_det

class RMALALayer(eqx.Module):
    """
    Diagonal Preconditioned MALA. 
    Adjusts step sizes per-parameter based on local sensitivity (identifiability).
    """
    log_posterior: Callable
    lambd: float
    num_mh_steps: int
    base_step_size: float
    apply_gradient_schedule: bool
    eps: float = 1e-6

    def __init__(self, log_posterior, lambd=1.0, num_mh_steps=1, step_size=1e-3, apply_gradient_schedule=False):
        self.log_posterior = log_posterior
        self.lambd = lambd
        self.num_mh_steps = num_mh_steps
        self.base_step_size = step_size
        self.apply_gradient_schedule = apply_gradient_schedule

    def get_interpolated_energy(self, x, condition, gradient_scale):
        if self.lambd < 0.001:
            return 0.5 * jnp.sum(x**2, axis=-1)
        lp = self.log_posterior(x, condition, gradient_scale)[0]
        prior = 0.5 * jnp.sum(x**2, axis=-1)
        return self.lambd * lp + (1.0 - self.lambd) * prior

    def get_geometry(self, x, condition, gradient_scale):
        def energy_single(x_s):
            return self.get_interpolated_energy(x_s[None, :], condition[0:1, :], gradient_scale)[0]
        
        val, grad = jax.vmap(jax.value_and_grad(energy_single))(x)
        # Empirical Fisher diagonal for preconditioning
        # stop_gradient here because we only want the metric for the step, 
        # not to differentiate through the metric calculation itself.
        metric_diag = jax.lax.stop_gradient(jnp.abs(grad) + self.eps)
        precond = 1.0 / metric_diag
        return val, grad, precond

    def __call__(self, x, condition, key, cond_emb=None, gradient_scale=0):
        # Initial energy for log_det calculation
        energy_init = self.get_interpolated_energy(x, condition, gradient_scale)
        
        # x_curr = jax.lax.stop_gradient(x)
        x_curr = x
        h = self.base_step_size / max(self.lambd, 1e-4)

        for _ in range(self.num_mh_steps):
            key, subkey = jr.split(key)
            e_c, g_c, p_c = self.get_geometry(x_curr, condition, gradient_scale)
            
            # Proposal
            noise = jr.normal(subkey, x_curr.shape)
            step_mean = x_curr - h * p_c * g_c
            step_std = jnp.sqrt(2 * h * p_c)
            x_prop = step_mean + step_std * noise
            
            # Reverse geometry
            e_p, g_p, p_p = self.get_geometry(x_prop, condition, gradient_scale)
            rev_step_mean = x_prop - h * p_p * g_p
            
            # Transition logs
            log_q_f = -0.5 * jnp.sum(((x_prop - step_mean)**2) / (2 * h * p_c), axis=-1)
            log_q_b = -0.5 * jnp.sum(((x_curr - rev_step_mean)**2) / (2 * h * p_p), axis=-1)
            
            log_alpha = (-e_p + e_c) + (log_q_b - log_q_f)
            u = jr.uniform(subkey, (x.shape[0],))
            accept = (jnp.log(u + 1e-10) < log_alpha)[:, None]
            x_curr = jnp.where(accept, x_prop, x_curr)

        x_out = x_curr
        energy_final = self.get_interpolated_energy(x_out, condition, gradient_scale)
        log_det = energy_final - energy_init
        
        if self.apply_gradient_schedule:
            log_det = gradient_schedule(log_det, gradient_scale)
        else:
            log_det = jax.lax.stop_gradient(log_det)
            
        return x_out, log_det


class BMCMCLayer(eqx.Module):
    """
    Barker MCMC layer: Robust to gradient noise and helpful for identifiability
    when gradients are poorly scaled.
    """
    log_posterior: Callable
    lambd: float
    num_mh_steps: int
    step_size: float
    apply_gradient_schedule: bool

    def __init__(self, log_posterior, lambd=1.0, num_mh_steps=1, step_size=1e-2, apply_gradient_schedule=False):
        self.log_posterior = log_posterior
        self.lambd = lambd
        self.num_mh_steps = num_mh_steps
        self.step_size = step_size
        self.apply_gradient_schedule = apply_gradient_schedule

    def get_interpolated_energy(self, x, condition, gradient_scale):
        lp = self.log_posterior(x, condition, gradient_scale)[0]
        prior = 0.5 * jnp.sum(x**2, axis=-1)
        return self.lambd * lp + (1.0 - self.lambd) * prior

    def __call__(self, x, condition, key, cond_emb=None, gradient_scale=0):
        energy_init = self.get_interpolated_energy(x, condition, gradient_scale)
        
        def energy_single(x_s):
            return self.get_interpolated_energy(x_s[None, :], condition[0:1, :], gradient_scale)[0]
        
        grad_fn = jax.vmap(jax.grad(energy_single))
        # x_curr = jax.lax.stop_gradient(x)
        x_curr = x
        
        for _ in range(self.num_mh_steps):
            key, subkey = jr.split(key)
            grad = jax.lax.stop_gradient(grad_fn(x_curr))
            
            z = jr.normal(subkey, x.shape) * self.step_size
            # Barker's rule
            prob_accept_comp = jax.nn.sigmoid(-grad * z)
            u = jr.uniform(subkey, x.shape)
            direction = jnp.where(u < prob_accept_comp, 1.0, -1.0)
            x_curr = x_curr + direction * z
            
        x_out = x_curr
        energy_final = self.get_interpolated_energy(x_out, condition, gradient_scale)
        log_det = energy_final - energy_init

        if self.apply_gradient_schedule:
            log_det = gradient_schedule(log_det, gradient_scale)
        else: 
            log_det = jax.lax.stop_gradient(log_det)
            
        return x_out, log_det

def gradient_schedule(x: jnp.ndarray, gradient_scale: float) -> jnp.ndarray:
    """
    Apply gradient scheduling in a JIT-friendly way.
    
    Uses jax.lax.cond to avoid creating computation graphs for both branches.
    """
    def full_grad(x):
        return x
    
    def no_grad(x):
        return jax.lax.stop_gradient(x)
    
    def partial_grad(x):
        stopped = jax.lax.stop_gradient(x)
        return gradient_scale * x + (1.0 - gradient_scale) * stopped
    
    # Use conditional to select the right computation path
    return jax.lax.cond(
        gradient_scale >= 0.999,
        full_grad,
        lambda x: jax.lax.cond(
            gradient_scale <= 0.001,
            no_grad,
            partial_grad,
            x
        ),
        x
    )

STOCHASTIC_LAYERS = (MALALayer, RMALALayer, MCMCLayer, BMCMCLayer)
class CondSNF(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    n_stochastic: List[int] = eqx.field(static=True)
    encoder: TimeSeriesEncoder
    transforms: List[eqx.Module]
    refinement: Optional[eqx.Module]
    log_posterior_fn: Callable = eqx.field(static=True)
    apply_gradient_schedule: bool
            
    def __init__(
        self, 
        dim: int, 
        cond_dim: int, 
        log_posterior_fn: Callable,
        key: jax.Array,
        n_coupling: int = 6, 
        n_stochastic: List[int] = [4],
        hidden_nf: int = 128, 
        num_hidden_nf: int = 2,
        encoded_latent_dim: int = 16, 
        hidden_rnn: int = 128,
        rnn_type: str = "lstm",
        flow_type: str = 'realnvp',
        mcmc_type='MALA',
        num_mh_steps: int = 1,
        mcmc_step_size: float = 1e-3,
        noise_std=0.01,
        apply_gradient_schedule=False
    ):
        self.dim = dim
        self.cond_dim = cond_dim
        if not n_stochastic:
            n_stochastic = []
        else:
            if n_stochastic == 0 or 0 in n_stochastic:
                if mcmc_type.upper() == 'MALA':
                    self.refinement = MALALayer(log_posterior_fn, 
                            lambd=1.0, num_langevin_steps=1, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                            apply_gradient_schedule=apply_gradient_schedule
                            )                       
                elif mcmc_type.upper() == 'RMALA':
                    self.refinement = RMALALayer(log_posterior_fn, 
                        lambd=1.0, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                        apply_gradient_schedule=apply_gradient_schedule
                        )
                elif mcmc_type.upper() == 'BMCMC':
                    self.refinement = BMCMCLayer(log_posterior_fn, 
                        lambd=1.0, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                        apply_gradient_schedule=apply_gradient_schedule
                        )
                else:
                    self.refinement = MCMCLayer(log_posterior_fn, 
                        lambd=1.0, num_mh_steps=num_mh_steps, noise_std=noise_std, 
                        apply_gradient_schedule=apply_gradient_schedule
                        )
            else:
                self.refinement = None
        self.n_stochastic = n_stochastic

        self.log_posterior_fn = log_posterior_fn
        self.apply_gradient_schedule = apply_gradient_schedule
        k_enc, k_flow = jax.random.split(key)
        self.encoder = TimeSeriesEncoder(
            cond_dim, hidden_rnn, encoded_latent_dim, k_enc, rnn=rnn_type
        )
        
        self.transforms = []
        keys = jax.random.split(k_flow, n_coupling + 2)

        flow_type = flow_type.lower()
        
        def get_mask(parity):
            return jnp.array([(i + parity) % 2 for i in range(dim)], dtype=jnp.float32)

        for i in range(n_coupling // 2):
            k_layer1, k_layer2, k_perm = jax.random.split(keys[i], 3)
            
            # Permutation
            self.transforms.append(RandomPermutation(dim, key=k_perm))
            
            # Coupling layers (RealNVP style)
            self.transforms.append(
                AffineCoupling(
                    dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask(i % 2), key=k_layer1
                )
            )
            self.transforms.append(
                AffineCoupling(
                    dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask((i + 1) % 2), key=k_layer2
                )
            )
            
            # Insert MALA layer at specified positions
            # Lambda goes from 0 (prior) to 1 (posterior)
            
            if len(n_stochastic) > 0:
                
                if i + 1 in n_stochastic:
                    lambd = 1 - (i + 1) / (n_coupling // 2)
                    print(f"Inserting {mcmc_type} layer at position {i+1} with lambda={lambd:.3f}")
                    if mcmc_type.upper() == 'MALA':
                        self.transforms.append(
                            MALALayer(log_posterior_fn, 
                            lambd=lambd, num_langevin_steps=1, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                            apply_gradient_schedule=apply_gradient_schedule
                            )
                        )
                    elif mcmc_type.upper() == 'RMALA':
                        self.transforms.append(
                            RMALALayer(log_posterior_fn, 
                            lambd=lambd, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                            apply_gradient_schedule=apply_gradient_schedule
                            )
                        )
                    elif mcmc_type.upper() == 'BMCMC':
                        self.transforms.append(
                            BMCMCLayer(log_posterior_fn, 
                            lambd=lambd, num_mh_steps=num_mh_steps, step_size=mcmc_step_size, 
                            apply_gradient_schedule=apply_gradient_schedule
                            )
                        )
                    else:
                        self.transforms.append(
                            MCMCLayer(log_posterior_fn, 
                            lambd=lambd, num_mh_steps=num_mh_steps, noise_std=noise_std, 
                            apply_gradient_schedule=apply_gradient_schedule
                            )
                        )
                
        if len(n_stochastic) > 0:
            mala_layers = [t for t in self.transforms if isinstance(t, STOCHASTIC_LAYERS)]
            print(f"Total {mcmc_type} layers at {n_stochastic}: {len(mala_layers)}")
        if self.refinement:
            print(f"num. of {mcmc_type} layers for refinement: {num_mh_steps}")

    def __flow_call__(
        self, z: jnp.ndarray, cond: jnp.ndarray, 
        cond_emb: jnp.ndarray, key: jax.Array, 
        reverse=False, gradient_scale: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Execute flow transformations."""
        logdet = jnp.zeros(z.shape[0])
        x = z
        
        # Split keys for stochastic layers
        num_stochastic = len([t for t in self.transforms if isinstance(t, STOCHASTIC_LAYERS)])
        if num_stochastic > 0:
            keys = jr.split(key, num_stochastic)
            key_idx = num_stochastic - 1 if reverse else 0
        else:
            keys = None
            key_idx = 0

        flow_list = self.transforms if not reverse else reversed(self.transforms)
        
        for t in flow_list:
            if isinstance(t, (ActNorm, RandomPermutation)):
                x, ldj = t(x, reverse=reverse)
            elif isinstance(t, STOCHASTIC_LAYERS):
                x, ldj = t(x, cond, keys[key_idx], cond_emb=cond_emb, gradient_scale=gradient_scale)
                key_idx += -1 if reverse else 1
            else:  # AffineCoupling
                x, ldj = t(x, cond_emb, reverse=reverse)
            
            logdet = logdet + ldj

        return x, logdet

    def __call__(
        self, x: jnp.ndarray, cond: jnp.ndarray, key: jax.Array, gradient_scale: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass: x -> z"""
        cond_emb = self.encoder(cond)
        z, logdet = self.__flow_call__(x, cond, cond_emb, key, gradient_scale=gradient_scale, reverse=False)
        return z, logdet   

    def inverse(
        self, z: jnp.ndarray, cond: jnp.ndarray, key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Inverse pass: z -> x"""
        cond_emb = self.encoder(cond)
        x, logdet = self.__flow_call__(z, cond, cond_emb, key, reverse=True)
        return x, logdet

    def log_prob(
        self, x: jnp.ndarray, cond: jnp.ndarray, key: jax.Array, gradient_scale: float = 0.0
    ) -> jnp.ndarray:
        """Compute log probability of x given condition."""
        z, logdet = self(x, cond, key, gradient_scale=gradient_scale)
        # Prior log prob (standard Gaussian)
        log_prob_z = -0.5 * (jnp.sum(z**2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return log_prob_z + logdet

    def sample(
        self, cond: jnp.ndarray, key: jax.Array, num_samples: int
    ) -> jnp.ndarray:
        """Generate samples from the learned distribution."""
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        
        # Repeat conditions for all samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)
        
        # Sample from prior
        z_key, s_key = jr.split(key, 2)
        z = jr.normal(z_key, (total_num, self.dim))
        
        # Transform through inverse flow
        samples, _ = self.__flow_call__(z, cond_rep, cond_emb_rep, s_key, reverse=True)
        
        # Reshape to (num_samples, batch_size, dim)
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

    def sample_with_MCMC(
        self, cond: jnp.ndarray, key: jax.Array, num_samples: int
    ) -> jnp.ndarray:
        """Generate samples from the learned distribution."""
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        
        # Repeat conditions for all samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)
        
        # Sample from prior
        z_key, s_key = jr.split(key, 2)
        z = jr.normal(z_key, (total_num, self.dim))
        
        # Transform through inverse flow
        samples = self.__flow_call__(z, cond_rep, cond_emb_rep, s_key, reverse=True)[0]
        if self.refinement:
            samples = self.refinement(samples, cond_rep, s_key, cond_emb_rep)[0]
        # Reshape to (num_samples, batch_size, dim)
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

def snf_loss(model, y: jnp.ndarray, true_params: jnp.ndarray, key: jax.Array, gradient_scale: float = 0.0):
    """Negative log likelihood loss."""
    log_prob = model.log_prob(true_params, y, key, gradient_scale)
    loss = -log_prob.mean()
    return loss

@eqx.filter_jit
def make_step(
    model, y: jnp.ndarray, true_params: jnp.ndarray, 
    key, opt_state, optimizer, filter_fn=None, gradient_scale: float = 0.0
):
    """Single training step."""
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return snf_loss(full_model, y, true_params, key, gradient_scale)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
    else:
        loss_fn = lambda m: snf_loss(m, y, true_params, key, gradient_scale)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss