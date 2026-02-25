import sys
import os
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
from numba import cuda
import math
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax

    jax.default_device('cpu')
import jax.random as jr
import equinox as eqx
from functools import partial
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from bayesinf.utils.generate_data import normalize_data, denormalize_data, mass_spring_ode, mass_spring_ode_l, mass_spring_ode_f, mass_spring_ode2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns 

STOCHASTIC_TYPES = {'MALA', 'RMALA', 'MCMC', 'BMCMC'}
FLOW_TYPES = {'REALNVP', 'NSF', 'FM', 'SBM', 'BNN', 'VAE', 'DDPM'}

multi_modal = False
force = not multi_modal
linear = False

if multi_modal:
    from bayesinf.utils.generate_data import (generate_mass_spring_data2, mass_spring_ode2)
    generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                     generate_mass_spring_data2)
else:
    if linear:
        from bayesinf.utils.generate_data import (generate_mass_spring_data_l, mass_spring_ode_l)
        generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                         generate_mass_spring_data_l)
    else:
        if force:
            from bayesinf.utils.generate_data import (generate_mass_spring_data_f, mass_spring_ode_f)
            generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                            generate_mass_spring_data_f)
        else:
            from bayesinf.utils.generate_data import (generate_mass_spring_data, mass_spring_ode)
            generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                            generate_mass_spring_data)

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

@partial(jax.jit, static_argnums=(3, 4))
def solve_dynamics_batch(y0_batch, theta_phys, t0, n_steps, ode_func):
    """
    Optimized: Processes all windows in a single parallel batch.
    y0_batch: (num_windows, dim)
    theta_phys: (dim,)
    """
    batched_ode = jax.vmap(ode_func, in_axes=(None, 0, None))
    # Define a step that handles all windows at once
    def batch_step(carry, _):
        states, t = carry
        # Vmap the RK4 logic over the states (windows)
        # theta_phys is a constant for all windows
        k1 = batched_ode(t, states, (theta_phys,))
        k2 = batched_ode(t + dt0/2, states + dt0/2 * k1, (theta_phys,))
        k3 = batched_ode(t + dt0/2, states + dt0/2 * k2, (theta_phys,))
        k4 = batched_ode(t + dt0, states + dt0 * k3, (theta_phys,))
        next_states = states + dt0/6 * (k1 + 2*k2 + 2*k3 + k4)
        return (next_states, t + dt0), next_states

    # lax.scan handles the temporal loop
    _, trajectories = jax.lax.scan(batch_step, (y0_batch, t0), None, length=n_steps)
    
    # Reshape from (n_steps, num_windows, dim) to (num_windows, n_steps, dim)
    trajectories = jnp.transpose(trajectories, (1, 0, 2))
    # Add initial states
    return jnp.concatenate([y0_batch[:, None, :], trajectories], axis=1)

@partial(jax.jit, static_argnums=(1, 2))
def prepare_windows(y_obs, window_size, num_windows):
    """
    Slices a continuous time-series into a batch of windows.
    
    Args:
        y_obs: The full observation array, shape (Total_Time, Dim)
        window_size: Length of each window (n_ahead + 1)
        num_windows: How many segments to extract
        
    Returns:
        A batch of windows, shape (num_windows, window_size, Dim)
    """
    total_len = y_obs.shape[0]
    
    # Calculate the step size to spread windows evenly across the whole trajectory
    # If num_windows=1, step doesn't matter. If > 1, we space them out.
    step = jax.lax.cond(
        num_windows > 1,
        lambda _: (total_len - window_size) // (num_windows - 1),
        lambda _: 0,
        None
    )
    
    def get_slice(i):
        # dynamic_slice(operand, start_indices, slice_sizes)
        # start_indices must be a tuple of integers
        start_idx = i * step
        return jax.lax.dynamic_slice(y_obs, (start_idx, 0), (window_size, y_obs.shape[1]))
    
    # Vmap allows us to grab all windows in one parallel operation
    return jax.vmap(get_slice)(jnp.arange(num_windows))

def compute_ensemble_likelihood_fast(theta, y_obs, n_ahead, ode_type, sigma, mean, std, mean_theta, std_theta):
    """
    Highly optimized ensemble likelihood.
    """
    num_windows = 4
    window_size = n_ahead + 1
    
    # 1. Selection & Denormalization (Lifted out of the solver)
    if ode_type == 'multi_modal':
        ode_func = mass_spring_ode2
        _m_y = jnp.stack([mean[...,0]/2, mean[...,1]/2, mean[...,0]/2, mean[...,1]/2], axis=-1)
        _s_y = jnp.stack([std[...,0]/2, std[...,1]/2, std[...,0]/2, std[...,1]/2], axis=-1)
    else:
        _m_y, _s_y = mean, std
        ode_func = {'linear': mass_spring_ode_l, 'force': mass_spring_ode_f}.get(ode_type, mass_spring_ode)

    # 2. Prepare Data (All windows)
    y_windows = prepare_windows(y_obs, window_size, num_windows) # (N, W, D)
    y0_batch = y_windows[:, 0, :]
    
    # Reconstruct 4D if multi_modal
    if ode_type == 'multi_modal':
        y0_batch = jnp.stack([y0_batch[:,0]/2, y0_batch[:,1]/2, y0_batch[:,0]/2, y0_batch[:,1]/2], axis=-1)
    
    # 3. Denormalize once
    y0_phys = (y0_batch * jnp.squeeze(_s_y)) + jnp.squeeze(_m_y)
    theta_phys = (theta * jnp.squeeze(std_theta)) + jnp.squeeze(mean_theta)
    
    # 4. Single Batch Solve (The speedup happens here)
    traj_phys = solve_dynamics_batch(y0_phys, theta_phys, 0.0, n_ahead, ode_func)
    
    # 5. Map Physics to Observations
    if ode_type == 'multi_modal':
        preds_phys = jnp.stack([
            traj_phys[..., 0] + traj_phys[..., 2], 
            traj_phys[..., 1] + traj_phys[..., 3]
        ], axis=-1)
    else:
        preds_phys = traj_phys
        
    # 6. Normalize and MSE
    preds = (preds_phys - jnp.squeeze(mean)) / jnp.squeeze(std)
    mse = jnp.mean(jnp.square(preds - y_windows))
    
    return -0.5 * mse * 0.05 / (sigma**2 + 1e-6)                                        

def generate_test_data(noise_std):
    test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
    test_D, mean, std = normalize_data(test_D)
    test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)
    return test_D, test_theta_p, mean, std, mean_theta, std_theta
    
def decide_params_from_path(path):
    filename = os.path.basename(path)
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    results = {
        "flow_type": parts[0],
        "n_stochastic": None,
        "mcmc_step_size": None,
        "num_mh_steps": None,
        "refine": None,
        "noise_std": 0,
        "data_set": None
    }
    # 2. Find the "Anchor": The part that looks like 'X' followed by digits (e.g., X10)
    idx_mh = -1
    for i, p in enumerate(parts):
        # Matches 'X' followed by one or more digits
        if re.fullmatch(r'X\d+', p):
            idx_mh = i
            break
    # CASE A: Stochastic/Refined model (Anchor found)
    if idx_mh != -1:
        data_set_str = "_".join(parts[idx_mh + 1:])
        results["num_mh_steps"] = int(parts[idx_mh][1:]) # Remove 'X' and convert
        results["mcmc_step_size"] = float(parts[idx_mh - 1])
        
        # Check if it's a "refine" case or a standard stochastic case
        if parts[1] == "refine":
            results["refine"] = parts[2] # e.g., 'MALA'
        else:
            # n_stochastic is everything between flow_type (index 0) 
            # and mcmc_step_size (index idx_mh - 1)
            n_parts = parts[1 : idx_mh - 1]
            if len(n_parts) == 1:
                results["n_stochastic"] = [int(n_parts[0])]
            else:
                results["n_stochastic"] = [int(x) for x in n_parts]
    else:
        # Pure Flow: everything after flow_type is data_set
        data_set_str = "_".join(parts[1:])
    
    results["data_set"] = data_set_str

    if "_noise_" in data_set_str:
        # Split by '_noise_', take the right side, then take the first part of that (before the next '_')
        noise_part = data_set_str.split("_noise_")[1]
        noise_val_str = noise_part.split("_")[0]
        try:
            results["noise_std"] = float(noise_val_str)
        except ValueError:
            results["noise_std"] = 0.0

    return results

def build_model(flow_type, n_stochastic, mcmc_step_size, num_mh_steps, refine, noise_std, mean, std, mean_theta, std_theta):
    def log_posterior_fn(theta, y_obs, gradient_scale=0.0):
        n_ahead = 5
        sigma = max(0.01, noise_std)
        
        # Determine ODE type string
        ode_type = 'multi_modal' if multi_modal else ('linear' if linear else ('force' if force else 'standard'))
        comp_traj = lambda t, y : compute_ensemble_likelihood_fast(t, y, n_ahead, ode_type, sigma, mean, std, mean_theta, std_theta)
        # --- Batching Logic ---
        if theta.ndim > 1:
            # Batch of parameters (from Flow)
            log_lik = jax.vmap(comp_traj)(theta, y_obs)
            log_prior = -0.5 * jnp.sum(theta**2, axis=-1)
        else:
            # Single parameter (from MCMC chain)
            log_lik = comp_traj(theta, y_obs)
            log_prior = -0.5 * jnp.sum(theta**2)

        log_lik = gradient_schedule(log_lik, gradient_scale)
        total_prob = log_lik + log_prior

        total_prob = jnp.where(jnp.isfinite(total_prob), total_prob, -1e8)
        return total_prob   
    # Initialize Model
    if flow_type.upper() == 'BNN':
        from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator
        model = BNNPosteriorEstimator(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                    key=model_key, noise_std=noise_std, mcmc_type=refine if refine else 'MCMC',
                                    mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                                    log_posterior_fn=log_posterior_fn if not refine else None)
    elif flow_type.upper() == 'VAE':
        from bayesinf.models.variational_autoencoder import cVAE
        model = cVAE(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                    key=model_key, mcmc_type=refine if refine else 'MCMC', noise_std=noise_std, 
                    mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                    log_posterior_fn=log_posterior_fn if not refine  else None)
    elif flow_type.upper() == 'DDPM':
        from bayesinf.models.denoising_diffusion_model import ConditionalDDPM
        model = ConditionalDDPM(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                hidden_sizes, key=model_key, mcmc_type=refine if refine else 'MCMC', noise_std=noise_std, 
                                mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                                log_posterior_fn=log_posterior_fn if not refine  else None)
    elif flow_type.upper() == 'SBM':
        from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel
        model = ConditionalScoreModel(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                    hidden_sizes, key=model_key, mcmc_type=refine if refine else 'MCMC', noise_std=noise_std, 
                                    mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                                    log_posterior_fn=log_posterior_fn if not refine  else None)
    elif flow_type.upper() == 'FM':
        from bayesinf.models.flow_matching import ConditionalFlowMatching
        model = ConditionalFlowMatching(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                        key=model_key, mcmc_type=refine if refine else 'MCMC', noise_std=noise_std, 
                                        mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                                        log_posterior_fn=log_posterior_fn if not refine  else None)
    elif flow_type.upper() in STOCHASTIC_TYPES:
        from bayesinf.models.MCMC_Flow import CondSNF
        model = CondSNF(param_dim, feature_dim, log_posterior_fn, key=model_key, n_coupling=n_coupling, 
                        n_stochastic=n_stochastic, hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                        encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, rnn_type='lstm', 
                        flow_type='realnvp', mcmc_type=flow_type, noise_std=noise_std, 
                        mcmc_step_size=mcmc_step_size, num_mh_steps=num_mh_steps, 
                        apply_gradient_schedule=False)  
    else:
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=n_coupling, 
                            hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                            encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                            rnn_type='lstm', flow_type=flow_type) 
    return model                                   

def check_noise_diversity(save_path_pt_list):
    
    check_diversity = []
    for path in save_path_pt_list:
        pt_name = os.path.basename(path)
        if 'noise' in pt_name:
            noise_part = pt_name.split("_noise_")[1]
            noise_val_str = noise_part.split("_")[0]
            try:
                check_diversity.append(float(noise_val_str))
            except ValueError:
                check_diversity.append(0.0)   
        else:
            check_diversity.append(0.0)  
    is_constant = all(math.isclose(x, check_diversity[0], rel_tol=1e-4, abs_tol=1e-4) for x in check_diversity)
    return not is_constant, check_diversity[0]

def load_model_and_sample(flow_types, save_path_pt_list):
    diverse, noise_std = check_noise_diversity(save_path_pt_list)
    if not diverse:
        test_D, test_theta_p, mean, std, mean_theta, std_theta = generate_test_data(noise_std)
    results_list = []
    for flow_type, save_path_pt in zip(flow_types, save_path_pt_list):
        print(os.path.basename(save_path_pt))
        results = {}
        params = decide_params_from_path(save_path_pt)
        n_stochastic, mcmc_step_size, num_mh_steps, n_stochastic, refine, noise_std = \
            params['n_stochastic'], params['mcmc_step_size'], params['num_mh_steps'], params['n_stochastic'], params['refine'], params['noise_std']
        if diverse:
            test_D, test_theta_p, mean, std, mean_theta, std_theta = generate_test_data(noise_std)
        model = build_model(flow_type, n_stochastic, mcmc_step_size, num_mh_steps, refine, noise_std, mean, std, mean_theta, std_theta)
        model = eqx.tree_deserialise_leaves(save_path_pt, model)
        inference_model = eqx.nn.inference_mode(model)
        if n_stochastic:
            if n_stochastic == 0 or 0 in n_stochastic or refine:
                test_theta_samples = inference_model.sample_with_MCMC(test_D, sample_key, n_samples)
            else:
                test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
        else:
            test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)

        test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])
        results['test_theta_p'] = test_theta_p
        results['test_theta_samples'] = test_theta_samples
        results['path'] = os.path.basename(save_path_pt)
        results_list.append(results)
    return results_list

def confidence_interval(samples, lower=0.025, upper=0.975):
    """Calculates the [2.5, 97.5] quantiles for the samples."""
    # samples shape: (n_test, n_samples, n_params)
    ci_low = jnp.quantile(samples, lower, axis=0)
    ci_high = jnp.quantile(samples, upper, axis=0)
    return ci_low, ci_high

def cicp_mciw(theta_approx, theta_test, ci_level):
    # Calculate Credible Intervals
    ci_low, ci_high = confidence_interval(theta_approx, lower=(1-ci_level)/2, upper=(1+ci_level)/2)
    
    # CICP: Proportion of true values within the CI
    # (n_test, n_params)
    within_interval = (theta_test > ci_low) & (theta_test < ci_high)
    cicp = jnp.mean(within_interval, axis=0)
    
    widths = ci_high - ci_low
    mciw = jnp.mean(widths, axis=0)
    
    return cicp, mciw

def calculate_metrics(test_samples, test_ground_truth, ci_level=0.95):
    theta_approx_mean = jnp.mean(test_samples, axis=0)

    rmse = jnp.sqrt(jnp.mean((theta_approx_mean - test_ground_truth) ** 2, axis=0))
        
    cicp, mciw = cicp_mciw(test_samples, test_ground_truth, ci_level)
    return {
            "RMSE": rmse.mean(),
            "CICP": cicp.mean(),
            "MCIW": mciw.mean()
        }

def calculate_min_dist(est, gt):
    """
    Computes the average distance between estimated centers and ground truth 
    using optimal transport (Hungarian Algorithm).
    est: (2, 2) array of estimated centers
    gt:  (2, 2) array of ground truth modes
    """
    dists = cdist(est, gt) # Computes distance between every pair
    row_ind, col_ind = linear_sum_assignment(dists) # Finds best matching
    return dists[row_ind, col_ind].mean() # Average error for this batch item

def calculate_2mode_metrics(test_samples, test_ground_truth, ci_level=0.95):
    """
    test_samples: (num_samples, batch_size, dim)
    test_ground_truth: (batch_size, dim) -> [k1, k3]
    """
    num_samples, batch_size, dim = test_samples.shape
    
    # Storage for per-batch results
    batch_rmse = []
    batch_mciw = []
    batch_cicp = []

    # Quantiles for credible intervals (e.g., 0.025 and 0.975 for 95% CI)
    lower_q = (1 - ci_level) / 2
    upper_q = 1 - lower_q

    for i in range(batch_size):
        # 1. Get data for this batch item
        current_samples = test_samples[:, i, :] # (S, D)
        k1_true, k3_true = test_ground_truth[i]
        gt_modes = np.array([[k1_true, k3_true], [k3_true, k1_true]]) # (2, D)

        # 2. Cluster samples into 2 modes
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42).fit(current_samples)
        labels = kmeans.labels_
        est_centers = kmeans.cluster_centers_ # (2, D)

        # 3. Match estimated clusters to ground truth modes (Hungarian Algorithm)
        dists = cdist(est_centers, gt_modes)
        est_idx, gt_idx = linear_sum_assignment(dists)
        
        # 4. Calculate RMSE for this batch (distance between matched centers)
        rmse = np.sqrt(np.mean(dists[est_idx, gt_idx]**2))
        batch_rmse.append(rmse)

        # 5. Calculate MCIW and CICP for each matched pair
        cluster_widths = []
        cluster_coverages = []

        for e_idx, g_idx in zip(est_idx, gt_idx):
            # Extract samples belonging to this specific cluster
            cluster_samples = current_samples[labels == e_idx]
            target_truth = gt_modes[g_idx]

            if len(cluster_samples) > 1:
                # Calculate Credible Interval per dimension
                lower_bounds = np.quantile(cluster_samples, lower_q, axis=0)
                upper_bounds = np.quantile(cluster_samples, upper_q, axis=0)
                
                # MCIW: Average width across dimensions for this cluster
                width = np.mean(upper_bounds - lower_bounds)
                cluster_widths.append(width)

                # CICP: Does the truth fall within [lower, upper] for ALL dimensions?
                covered = np.all((target_truth >= lower_bounds) & (target_truth <= upper_bounds))
                cluster_coverages.append(float(covered))
            else:
                cluster_widths.append(0.0)
                cluster_coverages.append(0.0)

        batch_mciw.append(np.mean(cluster_widths))
        batch_cicp.append(np.mean(cluster_coverages))

    # Aggregate over the whole dataset
    return {
        "RMSE": np.mean(batch_rmse),
        "MCIW": np.mean(batch_mciw),
        "CICP": np.mean(batch_cicp)
    }

def generate_comparison_grid_mm(model_names, results_list, indices=[0, 49, 99, 199]):
    n_rows_data = len(indices)
    n_cols = len(model_names)
    idx_k1, idx_k3 = 0, 1 
    
    # Mapping for 5-row grid (skipping index 2 for the gap)
    grid_rows = [0, 1, 3, 4] 

    # --- Step 1: Data Collection (Unchanged) ---
    plot_data_cache = [[None for _ in range(n_cols)] for _ in range(n_rows_data)]
    for col_idx, results in enumerate(results_list):
        samples = results['test_theta_samples']
        test_theta_p = results['test_theta_p']
        for row_idx, sample_id in enumerate(indices):
            plot_data_cache[row_idx][col_idx] = (samples[:, sample_id, idx_k1], samples[:, sample_id, idx_k3], 
                                                 test_theta_p[sample_id, idx_k1], test_theta_p[sample_id, idx_k3])

    # --- Step 2: Plotting with GridSpec ---
    fig = plt.figure(figsize=(n_cols * 5.5, (n_rows_data * 5) + 0.1))
    # 5 rows total, index 2 is the 0.4 height spacer
    gs = fig.add_gridspec(5, n_cols, height_ratios=[1, 1, 0.1, 1, 1])
    
    axes_list = [] # Store for drawing the line later

    for row_idx_data in range(n_rows_data):
        row_axes = []
        for col_idx in range(n_cols):
            # Create the subplot in the correct grid slot
            ax = fig.add_subplot(gs[grid_rows[row_idx_data], col_idx])
            row_axes.append(ax)
            est_x, est_y, gt_x, gt_y = plot_data_cache[row_idx_data][col_idx]
            
            sns.kdeplot(x=est_x, y=est_y, ax=ax, fill=True, cmap="Blues", levels=8, alpha=0.7)
            ax.scatter(est_x, est_y, c='darkblue', s=15, marker='h', alpha=0.6, edgecolors='none')
            ax.scatter(gt_x, gt_y, marker='X', s=350, color='orange', edgecolors='black', label='Truth A' if (row_idx_data==0 and col_idx==0) else "")
            ax.scatter(gt_y, gt_x, marker='X', s=350, color='green', edgecolors='black', label='Truth B (Symmetric)' if (row_idx_data==0 and col_idx==0) else "")

            if row_idx_data == 0: ax.set_title(model_names[col_idx], fontsize=32, fontweight='bold', pad=15)
            if col_idx == 0: ax.set_ylabel(param_names[1], fontsize=32)
            if row_idx_data == n_rows_data - 1: ax.set_xlabel(param_names[0], fontsize=32)
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
        axes_list.append(row_axes)

    # --- Step 3: Global Legend & Line ---
    handles, labels = axes_list[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=32, frameon=True, shadow=True)

    plt.tight_layout(rect=[0.05, 0.07, 1, 0.97])
    fig.canvas.draw() 
    # Correct midpoint calculation using the GridSpec rows
    pos_top = axes_list[1][0].get_position() # Bottom of Row 2
    pos_bot = axes_list[2][0].get_position() # Top of Row 3
    y_mid = (pos_top.y0 + pos_bot.y1) / 2

    separator = Line2D([0.05, 1.0], [y_mid, y_mid], transform=fig.transFigure, color='blue', linewidth=3, linestyle='--', alpha=0.8)
    fig.add_artist(separator)

    plt.savefig(os.path.join(figure_dir, "model_comparison_grid_multi_modal_snf.pdf"), dpi=200, bbox_inches='tight')
    plt.show()


def plot_refinement_bar_chart(rmse_mat, mciw_mat, cicp_mat):
    """
    Colorblind-friendly grouped bar chart with refined models shown 
    in darker shades and a 2x3 legend.
    """
    noise_scales = ['0.01', '0.02', '0.05', '0.1']
    metrics = {'RMSE': rmse_mat, 'MCIW': mciw_mat, 'CICP': cicp_mat}
    methods = ['RealNVP', 'FM', 'NSF']

    # --- Colorblind-Safe Palette (Okabe-Ito based) ---
    # Each pair is (Lighter/Base, Darker/Refined)
    colors = ['#1984c5', '#0097b2', '#835ecc']

    # --- Sizing ---
    TITLE_SIZE = 34
    LABEL_SIZE = 28
    TICK_SIZE = 24
    LEGEND_SIZE = 26

    fig, axes = plt.subplots(1, 3, figsize=(28, 12), dpi=100)
    
    x = np.arange(len(noise_scales))
    total_group_width = 0.82
    bar_width = total_group_width / 6 
    width = 0.24

    for i, (metric_name, data) in enumerate(metrics.items()):
        ax = axes[i]
        
        for j in range(3):
            # Positioning logic for the 3 pairs
            base_pos = x + (j - 1) * (bar_width * 2.1) - (bar_width / 2)
            offset = (j - 1) * width 

            # Plot Base (Lighter)
            ax.bar(x + offset - width/8, data[:, j], bar_width, alpha=0.5,
                   color=colors[j], label=f'{methods[j]}' if i == 0 else "")
            
            # Plot Refined (Darker)
            ax.bar(x + offset + width/8, data[:, j+3], bar_width, 
                   color=colors[j], label=f'{methods[j]} + Refine' if i == 0 else "")

        # --- Styling ---
        ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight='bold', pad=40)
        ax.set_xlabel('Noise Scale', fontsize=LABEL_SIZE, fontweight='bold', labelpad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(noise_scales, fontsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        
        # Clean look: remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Subtle horizontal grid for better value reading
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cbd5e1')

        if metric_name == 'CICP':
            ax.axhline(0.95, color='#e11d48', linestyle=':', linewidth=3, label='Target 0.95')
            ax.set_ylim(0, 1.1)

    # --- 2x3 Legend Reordering ---
    handles, labels = axes[0].get_legend_handles_labels()

    # ncol=3 means 3 columns, which forces 6 items into 2 rows
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.12), 
               ncol=3, 
               fontsize=LEGEND_SIZE, 
               frameon=False,
               columnspacing=5.0, # Large space between columns
               handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82) # Adjust to fit the 2-row legend
    plt.savefig(os.path.join(figure_dir, "snf_refine_comparison.pdf"), dpi=200, bbox_inches='tight')


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
results_dir = os.path.join(project_root, 'results')
figure_dir = os.path.join(project_root, 'evaluation')

dt0 = 0.05
t1 = 1.0
feature_dim = 2 
n_coupling = 8
hidden_dim_rnn = 64  # GRU hidden size
feature_latent_dim = 64
hidden_dim_time = 4
hidden_sizes = [64] * 2
param_dim = 2 if multi_modal else 3  # Number of physical parameters to infer (theta_p)
param_names = [r'$k_1$', r'$k_3$'] if multi_modal else [r'$m$', r'$k$', r'$c$']
model_key = jr.PRNGKey(42)
test_key = jr.PRNGKey(0)
test_key, sample_key = jr.split(test_key, 2)
test_size = 200
n_samples = 100 

# flow_types = ['BNN', 'VAE', 'DDPM', 'SBM', 'FM', 'RealNVP', 'NSF', 'RMALA']
# save_path_pt_list = [
#     'BNN_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'VAE_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'DDPM_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'SBM_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'FM_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'RealNVP_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'NSF_mass_spring_damper_0.05_1.0_multi_modal.pt',
#     'RMALA_4_0.001_X1_mass_spring_damper_0.05_1.0_multi_modal.pt'
# ]
# save_path_pt_list = [os.path.join(results_dir, p) for p in save_path_pt_list]
# results_list = load_model_and_sample(flow_types, save_path_pt_list)
# for r, results in enumerate(results_list):
#     metrics = calculate_2mode_metrics(results['test_theta_samples'], results['test_theta_p'])
#     results_list[r] = results_list[r] | metrics
save_path_pt_list = []
all_results_list = []
all_flow_types = []
rmses = []
mciws = []
cicps = []
noise_std_list = [0.01, 0.02, 0.05, 0.1]
for noise_std in noise_std_list:
    _flow_types = ['RealNVP', 'FM', 'NSF', 'RMALA', 'FM', 'NSF']
    _save_path_pt_list = [
    f'RealNVP_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt',
    f'FM_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt',
    f'NSF_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt',
    f'RMALA_0_0.001_X2_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt',
    f'FM_refine_RMALA_0.001_X2_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt',
    f'NSF_refine_RMALA_0.001_X2_mass_spring_damper_0.05_1.0_noise_{noise_std}_force.pt'
    ]

    _save_path_pt_list = [os.path.join(results_dir, p) for p in _save_path_pt_list]
    results_list = load_model_and_sample(_flow_types, _save_path_pt_list)
    
    rmse = []
    mciw = []
    cicp = []
    for r, results in enumerate(results_list):
        if 'multi_modal' in results['path']:
            metrics = calculate_2mode_metrics(results['test_theta_samples'], results['test_theta_p'])
        else:
            metrics = calculate_metrics(results['test_theta_samples'], results['test_theta_p'])
        rmse.append(metrics['RMSE'])
        mciw.append(metrics['MCIW'])
        cicp.append(metrics['CICP'])
        results_list[r] = results_list[r] | metrics
    rmses.append(np.array(rmse))
    mciws.append(np.array(mciw))
    cicps.append(np.array(cicp))   
    all_results_list += results_list

rmses = np.stack(rmses)
mciws = np.stack(mciws)
cicps = np.stack(cicps)