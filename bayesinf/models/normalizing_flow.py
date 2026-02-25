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
from typing import Any, List, Tuple, Optional
from functools import partial
from bayesinf.models.networks import TimeSeriesEncoder
from bayesinf.models.MCMC_Flow import MALALayer, RMALALayer, MCMCLayer, BMCMCLayer
STOCHASTIC_LAYERS = (MALALayer, RMALALayer, MCMCLayer, BMCMCLayer)

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
    idx_a: jnp.ndarray = eqx.field(static=True)
    idx_b: jnp.ndarray = eqx.field(static=True)
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

class MaskedLinear(eqx.nn.Linear):
    mask: jnp.ndarray

    def __init__(self, in_features, out_features, key):
        super().__init__(in_features, out_features, key=key)
        self.mask = jnp.ones((out_features, in_features))

    def set_mask(self, mask):
        # update mask, return new instance
        return eqx.tree_at(lambda l: l.mask, self, mask)

    def __call__(self, x):
        # x: (..., in_features)
        w_masked = self.weight * self.mask
        return x @ w_masked.T + self.bias


class MADE(eqx.Module):
    input_dim: int = eqx.field(static=True)
    hidden_sizes: List[int] = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    natural_ordering: bool = eqx.field(static=True)
    indices: List[int]
    net: List[MaskedLinear]

    def __init__(self, input_dim, hidden_sizes, cond_dim, natural_ordering, key):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.natural_ordering = natural_ordering
        self.hidden_sizes = hidden_sizes
        
        # Input layer size is Data + Condition
        hs = [input_dim + cond_dim] + hidden_sizes + [input_dim * 2]
        
        keys = jax.random.split(key, len(hs))
        self.net = []
        for i in range(len(hs) - 1):
            self.net.append(MaskedLinear(hs[i], hs[i+1], key=keys[i]))

        # Create masks and assign them
        self._create_masks(keys[-1])

    def _create_masks(self, key):
        D = self.input_dim
        C = self.cond_dim
        k1, k2 = jax.random.split(key)

        # 1. Degrees for Physical Parameters (1 to D)
        # These must respect autoregressive property
        if self.natural_ordering:
            degrees_x = jnp.arange(1, D + 1)
        else:
            degrees_x = jax.random.randint(k1, (D,), 1, D + 1)
        self.indices = jnp.argsort(degrees_x).tolist()

        # 2. Degrees for Condition (ALWAYS 0)
        # This ensures Hidden Units (deg >= 1) can ALWAYS see Condition (deg 0)
        degrees_c = jnp.zeros(C, dtype=jnp.int32)

        # Combined Input Degrees: [Params, Conditions]
        degrees_in = jnp.concatenate([degrees_x, degrees_c])
        degrees = [degrees_in]

        # 3. Hidden Layer Degrees
        for h in self.hidden_sizes:
            key, subkey = jax.random.split(key)
            # Hidden units range from 1 to D-1 (or D)
            degrees.append(jax.random.randint(subkey, (h,), 1, D))

        # 4. Output Degrees
        # We only reconstruct the parameters, so output degrees match degrees_x
        degrees.append(jnp.tile(degrees_x, 2)) # (D*2,)

        masks = []
        for l in range(len(degrees) - 1):
            d_in = degrees[l]
            d_out = degrees[l+1]
            
            # mask shape: (out, in)
            is_output_layer = (l == len(degrees) - 2)

            if is_output_layer:
                # Output layer cannot see itself (strictly >)
                mask = (d_out[:, None] > d_in[None, :]).astype(jnp.float32)
            else:
                # Hidden layers can see lower or equal degrees
                mask = (d_out[:, None] >= d_in[None, :]).astype(jnp.float32)
            masks.append(mask)

        # Assign masks to layers
        new_net = []
        for layer, m in zip(self.net, masks):
            new_net.append(layer.set_mask(m))
        self.net = new_net

    def __call__(self, x, cond=None):
        # Concatenate x (Params) and cond (Encoded Time Series)
        if cond is not None:
            h = jnp.concatenate([x, cond], axis=-1)
        else:
            h = x
            
        for i, layer in enumerate(self.net):
            h = layer(h)
            if i < len(self.net) - 1:
                h = jax.nn.relu(h)
        
        mu, log_scale = jnp.split(h, 2, axis=-1)
        return mu, log_scale


class MAFLayer(eqx.Module):
    dim: int = eqx.field(static=True)
    net: MADE

    def __init__(self, input_dim, hidden_sizes, cond_dim, natural_ordering, key):
        self.dim = input_dim
        self.net = MADE(input_dim, hidden_sizes, cond_dim, natural_ordering, key)

    def __call__(self, x, cond=None, reverse=False):
        if not reverse:
            # Forward: x -> z
            mu, log_scale = self.net(x, cond)
            log_scale = jnp.tanh(log_scale)
            # MAF Transformation: z = (x - mu) / scale
            z = (x - mu) * jnp.exp(-log_scale)
            
            # Log Det Jacobian = -sum(log(scale))
            ldj = -jnp.sum(log_scale, axis=-1)
            return z, ldj
        else:
            # Inverse: z -> x (Sequential)
            # z is the input here
            def _inverse_single(z_inp, c_inp):
                x_rec = jnp.zeros_like(z_inp)
                ldj = 0
                # We use a python loop because the dimension (3) is tiny.
                # Ideally use jax.lax.scan for large dimensions.
                def body_fun(val, idx):
                    x_current, ldj = val
                    mu, log_scale = self.net(x_current, c_inp)
                    log_scale = jnp.tanh(log_scale)
                    # it only depends on x_current[<i]
                    x_i = z_inp[idx] * jnp.exp(log_scale[idx]) + mu[idx]
                    ldj += log_scale[idx]
                    return (x_current.at[idx].set(x_i), ldj), None

                (x_rec, ldj), _ = jax.lax.scan(body_fun, (x_rec, ldj), jnp.array(self.net.indices))
                    
                return x_rec, ldj # LDJ not needed for sampling usually

            return jax.vmap(_inverse_single)(x, cond)


def searchsorted(bin_locations, inputs, eps=1e-6):
    # bin_locations: (..., num_bins+1)
    # inputs: (..., )
    # Returns index of bin (..., )
    # We just use jnp.searchsorted on the last axis.
    # jnp.searchsorted expects 1D sorted array usually, but supports batching if shapes match?
    # Actually jax.numpy.searchsorted is not batched in the same convenient way as torch.
    # We use vmap for robustness.
    
    return jax.vmap(lambda b, i: jnp.searchsorted(b, i, side='right') - 1)(bin_locations, inputs)


def rational_quadratic_spline(inputs, widths, heights, derivatives, inverse, 
                              tail_bound=3.0, min_bin_width=1e-3, 
                              min_bin_height=1e-3, min_derivative=1e-3):
    # inputs: (batch, dim)
    # widths, heights: (batch, dim, num_bins)
    # derivatives: (batch, dim, num_bins+1)
    
    num_bins = widths.shape[-1]
    
    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    # Outside mask handled by jnp.where at the end

    # Normalize params
    widths = jax.nn.softmax(widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.pad(cumwidths, ((0,0), (0,0), (1,0)), constant_values=0.0)
    cumwidths = (cumwidths - 0.5) * 2 * tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    heights = jax.nn.softmax(heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, ((0,0), (0,0), (1,0)), constant_values=0.0)
    cumheights = (cumheights - 0.5) * 2 * tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    derivatives = jax.nn.softplus(derivatives) + min_derivative

    # Search sorted
    # Flatten batch and dim for vmap application inside searchsorted helper or usage
    # or just align dimensions carefully.
    
    if inverse:
        bin_locations = cumheights
    else:
        bin_locations = cumwidths
        
    # We map over the last dimension (num_bins) logic
    # inputs: (B, D), locations: (B, D, K+1)
    # We need to find bin idx for each (b, d)
    
    # Helper to gather params based on bin_idx
    def _get_bin_params(b_locs, inp):
        # b_locs: (K+1,)
        # inp: scalar
        idx = jnp.searchsorted(b_locs, inp, side='right') - 1
        return jnp.clip(idx, 0, num_bins - 1)

    # Vectorize over batch and dim
    bin_idx = jax.vmap(jax.vmap(_get_bin_params))(bin_locations, inputs) # (B, D)
    
    # Gather using take_along_axis
    # Need to expand bin_idx to (B, D, 1) for take_along_axis
    idx_expanded = bin_idx[..., None]
    
    input_cumwidths = jnp.take_along_axis(cumwidths, idx_expanded, axis=-1).squeeze(-1)
    input_bin_widths = jnp.take_along_axis(widths, idx_expanded, axis=-1).squeeze(-1)
    input_cumheights = jnp.take_along_axis(cumheights, idx_expanded, axis=-1).squeeze(-1)
    input_bin_heights = jnp.take_along_axis(heights, idx_expanded, axis=-1).squeeze(-1)
    
    input_delta = input_bin_heights / input_bin_widths
    
    input_derivatives = jnp.take_along_axis(derivatives, idx_expanded, axis=-1).squeeze(-1)
    input_derivatives_plus_one = jnp.take_along_axis(derivatives, idx_expanded + 1, axis=-1).squeeze(-1)

    # Calculations
    if inverse:
        a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_bin_heights * (input_delta - input_derivatives)
        b = input_bin_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        
        discriminant = jnp.square(b) - 4 * a * c
        # Avoid NaN in sqrt if discriminant slightly neg due to float error
        discriminant = jnp.maximum(discriminant, 0.0) 
        
        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs_inner = root * input_bin_widths + input_cumwidths
        
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
        )
        derivative_numerator = jnp.square(input_delta) * (
                input_derivatives_plus_one * jnp.square(root)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * jnp.square(1 - root)
        )
        logabsdet_inner = -jnp.log(derivative_numerator) + 2 * jnp.log(denominator)

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        
        numerator = input_bin_heights * (
                input_delta * jnp.square(theta) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
        )
        outputs_inner = input_cumheights + numerator / denominator
        
        derivative_numerator = jnp.square(input_delta) * (
                input_derivatives_plus_one * jnp.square(theta)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * jnp.square(1 - theta)
        )
        logabsdet_inner = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)

    # Apply bounds
    outputs = jnp.where(inside_mask, outputs_inner, inputs)
    logabsdet = jnp.where(inside_mask, logabsdet_inner, jnp.zeros_like(inputs))
    
    return outputs, jnp.sum(logabsdet, axis=-1)


class SplineCoupling(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    idx_a: jnp.ndarray = eqx.field(static=True)
    idx_b: jnp.ndarray = eqx.field(static=True)
    net: eqx.nn.Sequential
    tail_bound: float = 3.0
    
    def __init__(self, input_dim, hidden_sizes, cond_dim, mask, key, num_bins=8, tail_bound=3.0):
        self.dim = input_dim
        self.cond_dim = cond_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.idx_a = jnp.where(mask)[0]  # Indices where mask=1 (conditioner input)
        self.idx_b = jnp.where(1 - mask)[0] # Indices where mask=0 (transformed)
        
        in_dim = int(jnp.sum(mask)) + cond_dim
        out_dim = input_dim - int(jnp.sum(mask))
        param_dim = out_dim * (3 * num_bins + 1)
        
        layers = []
        curr_dim = in_dim
        for h in hidden_sizes:
            k1, key = jax.random.split(key)
            layers.append(eqx.nn.Linear(curr_dim, h, key=k1))
            layers.append(eqx.nn.Lambda(jax.nn.relu))
            curr_dim = h
        
        k2, key = jax.random.split(key)
        final = eqx.nn.Linear(curr_dim, param_dim, key=k2)
        
        # Init final layer to zero
        zeros_w = jnp.zeros_like(final.weight)
        zeros_b = jnp.zeros_like(final.bias)
        final = eqx.tree_at(lambda l: (l.weight, l.bias), final, (zeros_w, zeros_b))
        layers.append(final)
        
        self.net = eqx.nn.Sequential(layers)

    def __call__(self, x, cond, reverse=False):

        # Split
        x_a = x[..., self.idx_a]
        x_b = x[..., self.idx_b]
        
        inp = jnp.concatenate([x_a, cond], axis=-1)
        params = jax.vmap(self.net)(inp)
        
        # Reshape params
        batch_size = x.shape[0]
        out_dim = x_b.shape[-1]
        params = params.reshape(batch_size, out_dim, 3 * self.num_bins + 1)
        
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        z_b, ldj = rational_quadratic_spline(
            x_b, widths, heights, derivatives, reverse, 
            tail_bound=self.tail_bound
        )
        
        z = jnp.zeros_like(x)
        z = z.at[..., self.idx_a].set(x_a)
        z = z.at[..., self.idx_b].set(z_b)
        
        return z, ldj


class CondNormFlow(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    encoder: TimeSeriesEncoder
    transforms: List[eqx.Module]
    prior_mean: jnp.ndarray
    prior_cov: jnp.ndarray
    refinement: Optional[eqx.Module] 

    def __init__(
        self, 
        dim: int, 
        cond_dim: int, 
        key: jax.Array,
        n_coupling: int = 6, 
        hidden_nf: int = 128, 
        num_hidden_nf: int = 2,
        encoded_latent_dim: int = 16, 
        hidden_rnn: int = 128,
        rnn_type: str = "lstm",
        flow_type: str = 'realnvp',
        log_posterior_fn=None,
        mcmc_type='MALA', num_mh_steps: int = 1, mcmc_step_size: float = 1e-3, noise_std=0.01
    ):
        self.dim = dim
        self.cond_dim = cond_dim
        self.prior_mean = jnp.zeros(dim)
        self.prior_cov = jnp.eye(dim)
        
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
            if flow_type == 'nsf':
                self.transforms.append(
                    SplineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, get_mask(i % 2),
                                   key=k_layer1, num_bins=4)
                )
                self.transforms.append(
                    SplineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, get_mask((i + 1) % 2),
                                   key=k_layer2, num_bins=4)
                )
            elif flow_type == 'maf':
                self.transforms.append(
                    MAFLayer(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, False, key=k_layer1)
                )
            else:
                # RealNVP
                self.transforms.append(
                    AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask(i % 2), key=k_layer1)
                )
                self.transforms.append(
                    AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask((i + 1) % 2), key=k_layer2)
                )

        # Final ActNorm
        self.transforms.append(ActNorm(dim))
        
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

    def __flow_call__(self, z: jnp.ndarray, cond_emb:jnp.ndarray, reverse=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(z.shape[:-1])
        x = z

        flow_list = self.transforms if not reverse else reversed(self.transforms)
        for t in flow_list:
            if isinstance(t, (ActNorm, RandomPermutation)):
                x, ldj = t(x, reverse=reverse)
            else:
                x, ldj = t(x, cond_emb, reverse=reverse)
            logdet = logdet + ldj

        return x, logdet

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # cond: raw time series (Batch, T, C)
        # Encode condition
        cond_emb = self.encoder(cond)
        z, logdet = self.__flow_call__(x, cond_emb, reverse=False)
        return z, logdet   

    def inverse(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cond_emb = self.encoder(cond)
        x, logdet = self.__flow_call__(z, cond_emb, reverse=True)
        return x, logdet

    def log_prob(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        z, logdet = self(x, cond)
        # Prior log prob (Gaussian)
        # -0.5 * (z^T z + D * log(2pi))
        log_prob_z = -0.5 * (jnp.sum(z**2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return log_prob_z + logdet

    def sample(self, cond: jnp.ndarray, key: jax.Array, num_samples: int) -> jnp.ndarray:
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        cond_rep = jnp.repeat(cond_emb, num_samples, axis=0)

        z = jr.normal(key, (total_num, self.dim))
        samples = self.__flow_call__(z, cond_rep, reverse=True)[0]
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

    def sample_with_MCMC(self, cond: jnp.ndarray, key: jax.Array, num_samples: int) -> jnp.ndarray:
        cond_emb = self.encoder(cond)
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        cond_emb_rep = jnp.repeat(cond_emb, num_samples, axis=0)
        cond_rep = jnp.repeat(cond, num_samples, axis=0)

        z = jr.normal(key, (total_num, self.dim))
        samples = self.__flow_call__(z, cond_emb_rep, reverse=True)[0]
        if self.refinement:
            s_key = jr.split(key, 2)[1]
            samples = self.refinement(samples, cond_rep, s_key, cond_emb_rep)[0]
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples


def cnf_loss(model, y: jnp.ndarray, true_params: jnp.ndarray):
    ml_loss = model.log_prob(true_params, y)
    loss = - ml_loss.mean()
    return loss


@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state: Any, optimizer: optax.GradientTransformation, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return cnf_loss(full_model, y, true_params)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
    else:
        loss_fn = lambda m: cnf_loss(m, y, true_params)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Example usage snippet
if __name__ == "__main__":

    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..')
    sys.path.append(project_root)
    from utils.generate_data import generate_mass_spring_data

    flow_type = 'maf' if len(sys.argv) == 1 else sys.argv[1]
    dt0 = 0.01
    t1 = 1.0
    noise_std = 0.0
    generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                            generate_mass_spring_data)

    feature_dim = 2  # Input features per time step
    hidden_dim_rnn = 64  # GRU hidden size
    hidden_dim = 64
    latent_dim = 16  # Encoded feature size
    param_dim = 3  # Number of physical parameters to infer (theta_p)
    param_names = [r'$m$', r'$k$', r'$c$']

    test_key = jr.PRNGKey(0)
    test_key, sample_key, noise_key = jr.split(test_key, 3)
    test_size = 200
    n_samples = 50
    test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)

    model_key = jr.PRNGKey(42)

    # Initialize Model
    model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, hidden_nf=hidden_dim, num_hidden_nf=2,
                         encoded_latent_dim=latent_dim, hidden_rnn=hidden_dim_rnn, rnn_type='lstm', flow_type=flow_type)


    def check_invertibility(model, x, cond, tol=1e-5):
        """
        Verifies x == inverse(forward(x)) and z == forward(inverse(z))
        """
        print(f"--- Checking {type(model).__name__} ---")
    
        # 1. Forward -> Inverse reconstruction
        z, ldj_fwd = model(x, cond)
        x_rec, ldj_inv = model.inverse(z, cond)
    
        # Check Data Reconstruction
        diff_x = jnp.max(jnp.abs(x - x_rec))
        is_invertible_x = diff_x < tol
    
        # Check LogDet Consistency: LDJ_forward + LDJ_inverse should be ~ 0
        diff_ldj = jnp.max(jnp.abs(ldj_fwd + ldj_inv))
    
        print(f"Max Reconstruction Error (x -> z -> x): {diff_x:.2e}")
        print(f"Max LogDet Error (LDJ_fwd + LDJ_inv):   {diff_ldj:.2e}")
    
        if not is_invertible_x:
            print("❌ Invertibility check FAILED (Tolerance exceeded)")
        else:
            print("✅ Invertibility check PASSED")
    
        return is_invertible_x
    
    
    check_invertibility(model, test_theta_p, test_D, tol=1e-5)