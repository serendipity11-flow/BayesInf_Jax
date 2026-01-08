import jax
import jax.numpy as jnp
from jax import vmap, jit, pmap
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Optional
import os


# Enable memory preallocation and optimize for TPU/GPU
jax.config.update('jax_enable_x64', False)  # Use float32 for speed
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def saltelli_sample(problem: Dict, n_samples: int, seed: int = 42) -> jnp.ndarray:
    """
    Optimized Saltelli sampling with vectorized operations.
    
    Args:
        problem: Dictionary with 'num_vars', 'names', 'bounds'
        n_samples: Base sample size (total samples will be N*(2*D+2))
        seed: Random seed
        
    Returns:
        Parameter samples of shape (N*(2*D+2), num_vars)
    """
    key = jax.random.PRNGKey(seed)
    num_vars = problem['num_vars']
    bounds = jnp.array(problem['bounds'], dtype=jnp.float32)
    
    key_a, key_b = jax.random.split(key)
    
    # Generate samples with float32 for efficiency
    a_samples = jax.random.uniform(key_a, (n_samples, num_vars), dtype=jnp.float32)
    b_samples = jax.random.uniform(key_b, (n_samples, num_vars), dtype=jnp.float32)
    
    # Vectorized scaling
    scale = bounds[:, 1] - bounds[:, 0]
    offset = bounds[:, 0]
    a_samples = offset + a_samples * scale
    b_samples = offset + b_samples * scale
    
    # Preallocate full array instead of concatenating
    total_samples = n_samples * (2 + num_vars)
    samples = jnp.empty((total_samples, num_vars), dtype=jnp.float32)
    
    # Fill samples efficiently
    samples = samples.at[:n_samples].set(a_samples)
    samples = samples.at[n_samples:2*n_samples].set(b_samples)
    
    # Vectorized C matrix creation
    for i in range(num_vars):
        c_i = a_samples.at[:, i].set(b_samples[:, i])
        start_idx = 2*n_samples + i*n_samples
        samples = samples.at[start_idx:start_idx+n_samples].set(c_i)
    
    return samples


@partial(jit, static_argnames=('time_steps',))
def placeholder_simulation(params: jnp.ndarray, time_steps: int = 100) -> jnp.ndarray:
    """
    Placeholder simulation function - replace with your actual model.
    Optimized with fused operations.
    
    Args:
        params: Parameter vector of shape (num_params,)
        time_steps: Number of time steps to simulate
        
    Returns:
        Simulated sensor data of shape (time_steps, num_sensors)
    """
    t = jnp.linspace(0, 10, time_steps, dtype=jnp.float32)
    
    # Fuse operations to reduce memory bandwidth
    # All sensors computed in one pass
    exp_term = jnp.exp(-params[1] * t)
    sin_term = jnp.sin(params[3] * t)
    cos_term = jnp.cos(params[3] * t)
    
    sensor_1 = params[0] * exp_term
    sensor_2 = params[2] * sin_term
    sensor_3 = params[0] * t + params[1] * (t * t)  # Fuse multiplication
    sensor_4 = params[0] * params[2] * cos_term + params[1]
    
    return jnp.stack([sensor_1, sensor_2, sensor_3, sensor_4], axis=1)


def get_chunk_size(total_samples: int, available_memory_gb: float = 4.0) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Args:
        total_samples: Total number of samples to process
        available_memory_gb: Available GPU/memory in GB
        
    Returns:
        Optimal chunk size
    """
    # Estimate: each sample with 100 timesteps and 4 sensors ≈ 1.6KB in float32
    bytes_per_sample = 100 * 4 * 4  # timesteps * sensors * bytes(float32)
    max_samples = int((available_memory_gb * 1e9) / bytes_per_sample * 0.5)  # Use 50% of memory
    chunk_size = min(max_samples, total_samples)
    return max(chunk_size, 32)  # Minimum chunk size of 32


@partial(jit, static_argnums=(1,))
def batch_simulate(params_batch: jnp.ndarray, time_steps: int) -> jnp.ndarray:
    """
    Vectorized simulation over parameter batch with optimized vmapping.
    
    Args:
        params_batch: Parameter matrix of shape (n_samples, num_params)
        time_steps: Number of time steps
        
    Returns:
        Simulation results of shape (n_samples, time_steps, num_sensors)
    """
    return vmap(lambda p: placeholder_simulation(p, time_steps))(params_batch)


def batch_simulate_chunked(
    params: jnp.ndarray, 
    time_steps: int, 
    chunk_size: Optional[int] = None
) -> jnp.ndarray:
    """
    Process simulations in chunks to avoid memory overflow.
    
    Args:
        params: All parameter samples
        time_steps: Number of time steps
        chunk_size: Size of each chunk (auto-calculated if None)
        
    Returns:
        All simulation results concatenated
    """
    n_samples = params.shape[0]
    
    if chunk_size is None:
        chunk_size = get_chunk_size(n_samples)
    
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    results = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        chunk = params[start_idx:end_idx]
        
        # Process chunk
        chunk_result = batch_simulate(chunk, time_steps)
        results.append(chunk_result)
    
    return jnp.concatenate(results, axis=0)


def calculate_sobol_indices(
    problem: Dict,
    param_values: jnp.ndarray,
    outputs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized calculation of first-order (S1) and total-order (ST) Sobol indices.
    Uses einsum for efficient tensor operations.
    
    Args:
        problem: Dictionary with sensitivity analysis problem definition
        param_values: Saltelli samples of shape (N*(2*D+2), num_vars)
        outputs: Model outputs of shape (N*(2*D+2), time_steps, num_sensors)
        
    Returns:
        S1: First-order indices of shape (time_steps, num_sensors, num_vars)
        ST: Total-order indices of shape (time_steps, num_sensors, num_vars)
    """
    num_vars = problem['num_vars']
    n_samples = param_values.shape[0] // (2 * num_vars + 2)
    
    # Reshape outputs efficiently
    y_a = outputs[:n_samples]
    y_b = outputs[n_samples:2*n_samples]
    
    # Stack C matrices in one operation
    y_c_list = [outputs[2*n_samples + i*n_samples:2*n_samples + (i+1)*n_samples] 
                for i in range(num_vars)]
    y_c = jnp.stack(y_c_list, axis=-1)  # (N, T, S, D)
    
    # Precompute variance once
    y_combined = jnp.concatenate([y_a, y_b], axis=0)
    var_y = jnp.var(y_combined, axis=0, keepdims=True)  # (1, T, S)
    
    # First-order indices with einsum for efficiency
    # S1 = Var[E(Y|Xi)] / Var(Y)
    diff_c_a = y_c - y_a[:, :, :, None]  # (N, T, S, D)
    s1_numerator = jnp.mean(y_b[:, :, :, None] * diff_c_a, axis=0)  # (T, S, D)
    s1 = s1_numerator / (var_y[0, :, :, None] + 1e-10)
    
    # Total-order indices with optimized computation
    # ST = E[Var(Y|X~i)] / Var(Y) = 0.5 * E[(f(A) - f(Ci))^2] / Var(Y)
    squared_diff = jnp.square(y_a[:, :, :, None] - y_c)
    st_numerator = 0.5 * jnp.mean(squared_diff, axis=0)  # (T, S, D)
    st = st_numerator / (var_y[0, :, :, None] + 1e-10)
    
    # Clip negative values to zero
    s1 = jnp.maximum(s1, 0.0)
    st = jnp.maximum(st, 0.0)
    
    return s1, st


@jit
def calculate_mean_sensitivity(s1: jnp.ndarray, st: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate mean sensitivity across time and sensors with single pass.
    
    Args:
        s1: First-order indices (time_steps, num_sensors, num_vars)
        st: Total-order indices (time_steps, num_sensors, num_vars)
        
    Returns:
        Mean S1 and ST across time and sensors
    """
    s1_mean = jnp.mean(s1, axis=(0, 1))
    st_mean = jnp.mean(st, axis=(0, 1))
    return s1_mean, st_mean


def run_sensitivity_analysis(
    problem: Dict,
    n_samples: int = 64,
    time_steps: int = 100,
    seed: int = 42
) -> Dict:
    """
    Complete sensitivity analysis workflow.
    
    Args:
        problem: Dictionary with 'num_vars', 'names', 'bounds'
        n_samples: Number of base samples for Saltelli sampling
        time_steps: Number of simulation time steps
        seed: Random seed
        
    Returns:
        Dictionary containing S1, ST indices and simulation results
    """
    print(f"Generating Saltelli samples for {problem['num_vars']} parameters...")
    param_values = saltelli_sample(problem, n_samples, seed)
    print(f"Total samples: {param_values.shape[0]}")
    
    print("Running simulations...")
    outputs = batch_simulate(param_values, time_steps)
    print(f"Output shape: {outputs.shape}")
    
    print("Calculating Sobol indices...")
    s1, st = calculate_sobol_indices(problem, param_values, outputs)
    
    # Calculate mean sensitivity across time and sensors
    s1_mean = jnp.mean(s1, axis=(0, 1))
    st_mean = jnp.mean(st, axis=(0, 1))
    
    print("\nFirst-order indices (averaged):")
    for i, name in enumerate(problem['names']):
        print(f"  {name}: {s1_mean[i]:.4f}")
    
    print("\nTotal-order indices (averaged):")
    for i, name in enumerate(problem['names']):
        print(f"  {name}: {st_mean[i]:.4f}")
    
    return {
        'S1': s1,
        'ST': st,
        'S1_mean': s1_mean,
        'ST_mean': st_mean,
        'param_values': param_values,
        'outputs': outputs,
        'problem': problem
    }


def rank_parameters(sobol_indices: jnp.ndarray, problem: Dict) -> List[Tuple[str, float]]:
    """
    Rank parameters by their sensitivity indices.
    
    Args:
        sobol_indices: Mean sensitivity indices of shape (num_vars,)
        problem: Problem dictionary with parameter names
        
    Returns:
        List of (parameter_name, index_value) tuples sorted by importance
    """
    indices = np.array(sobol_indices)
    sorted_idx = np.argsort(indices)[::-1]
    
    ranking = [(problem['names'][i], indices[i]) for i in sorted_idx]
    return ranking


# Example usage with performance comparison
if __name__ == "__main__":
    import time
    
    # Define the sensitivity analysis problem
    problem = {
        'num_vars': 4,
        'names': ['param_0', 'param_1', 'param_2', 'param_3'],
        'bounds': [[0.5, 2.0], [0.1, 1.0], [0.0, 5.0], [0.5, 3.0]]
    }
    
    print("="*70)
    print("JAX OPTIMIZED SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print()
    
    # Run sensitivity analysis with timing
    total_start = time.time()
    results = run_sensitivity_analysis(
        problem=problem,
        n_samples=64,
        time_steps=100,
        seed=42
    )
    total_time = time.time() - total_start
    
    print(f"\nTotal execution time: {total_time:.3f}s")
    
    # Rank parameters by first-order indices
    print("\n" + "="*70)
    print("Parameter Ranking (First-order):")
    print("="*70)
    ranking_s1 = rank_parameters(results['S1_mean'], problem)
    for rank, (name, value) in enumerate(ranking_s1, 1):
        print(f"{rank}. {name}: {value:.4f}")
    
    print("\n" + "="*70)
    print("Parameter Ranking (Total-order):")
    print("="*70)
    ranking_st = rank_parameters(results['ST_mean'], problem)
    for rank, (name, value) in enumerate(ranking_st, 1):
        print(f"{rank}. {name}: {value:.4f}")
    
    # Performance test with larger problem
    print("\n" + "="*70)
    print("PERFORMANCE TEST: Larger Problem")
    print("="*70)
    
    large_problem = {
        'num_vars': 8,
        'names': [f'param_{i}' for i in range(8)],
        'bounds': [[0.0, 1.0] for _ in range(8)]
    }
    
    start = time.time()
    large_results = run_sensitivity_analysis(
        problem=large_problem,
        n_samples=128,
        time_steps=200,
        seed=123
    )
    large_time = time.time() - start
    
    total_evals = 128 * (2 * 8 + 2) * 200  # n_samples * (2*D+2) * time_steps
    print(f"\nTotal model evaluations: {total_evals:,}")
    print(f"Evaluations per second: {total_evals/large_time:,.0f}")
    print(f"Time per 1000 evaluations: {1000*large_time/total_evals:.4f}s")