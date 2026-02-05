import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
import jax.random as jr
import equinox as eqx
import jax
from functools import partial
from sklearn.metrics import r2_score
import jax.numpy as jnp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from bayesinf.utils.generate_data import normalize_data, denormalize_data

from bayesinf.utils.generate_data import (generate_mass_spring_data_f, mass_spring_ode_f)
generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                generate_mass_spring_data_f)

def create_model(flow_type):
    # Initialize Model
    if flow_type.upper() == 'BNN':
        from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator
        model = BNNPosteriorEstimator(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                    key=model_key)
    elif flow_type.upper() == 'VAE':
        from bayesinf.models.variational_autoencoder import cVAE
        model = cVAE(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                    key=model_key)
    elif flow_type.upper() == 'DDPM':
        from bayesinf.models.denoising_diffusion_model import ConditionalDDPM
        model = ConditionalDDPM(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                hidden_sizes, key=model_key)
    elif flow_type.upper() == 'SBM':
        from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel
        model = ConditionalScoreModel(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                    hidden_sizes, key=model_key)
    elif flow_type.upper() == 'FM':
        from bayesinf.models.flow_matching import ConditionalFlowMatching
        model = ConditionalFlowMatching(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                        key=model_key)
    elif flow_type.upper() == 'MALA':
        from bayesinf.models.MCMC_Flow import CondSNF

        def dynamics_fn(t, state_norm, theta_norm):
            state = denormalize_data(state_norm, jnp.squeeze(mean), jnp.squeeze(std))
            theta = denormalize_data(theta_norm, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
            deriv = jax.vmap(lambda _t, _s: mass_spring_ode_f(_t, _s, (theta,)))(t, state)
            deriv_norm = normalize_data(deriv, jnp.squeeze(mean), jnp.squeeze(std))
            return deriv_norm

        def log_posterior_fn(x, y):
            num_steps = int(t1/dt0)
            ts = jnp.linspace(0, t1-dt0, num_steps)
            if x.ndim > 1:
                states = y[:, 1:] - y[:, :-1]
            else:
                states = y[1:] - y[:-1]
            if x.ndim > 1:
                y0 = jnp.stack([y[:, 0, 0]/2, jnp.zeros((y.shape[0],)), y[:, 0, 0]/2, jnp.zeros((y.shape[0],))], axis=-1)
                pred = jax.vmap(dynamics_fn, in_axes=(0, 0, None))(y0, x, ts)
            else:
                y0 = jnp.array([y[0, 0]/2, 0, y[0, 0]/2, 0])
                pred = dynamics_fn(y0, x, ts)
            residual = pred - states
            
            std = max(0.01, noise_std)
            if x.ndim > 1:
                log_likelihood = -0.5 * jnp.sum(residual ** 2, axis=(1, 2)) / (num_steps * std ** 2)
                log_prior = -0.5 * jnp.sum(x**2, axis=-1)
            else:
                log_likelihood = -0.5 * jnp.sum(residual ** 2) / (num_steps * std ** 2)
                log_prior = -0.5 * jnp.sum(x ** 2)
            return log_likelihood + log_prior

        model = CondSNF(param_dim, feature_dim, log_posterior_fn, key=model_key, n_coupling=8, 
                        hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                        encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                        rnn_type='lstm', flow_type='realnvp', metr_steps_per_block=3,
                        lang_steps=1, stepsize=5e-3)          
    else:
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, 
                            hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                            encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                            rnn_type='lstm', flow_type=flow_type)  
    return model     

def confidence_interval(samples, lower=0.025, upper=0.975):
    """Calculates the [2.5, 97.5] quantiles for the samples."""
    # samples shape: (n_test, n_samples, n_params)
    ci_low = jnp.quantile(samples, lower, axis=0)
    ci_high = jnp.quantile(samples, upper, axis=0)
    return ci_low, ci_high

def cicp_mciw(theta_approx, theta_test):
    # Calculate Credible Intervals
    ci_low, ci_high = confidence_interval(theta_approx)
    
    # CICP: Proportion of true values within the CI
    # (n_test, n_params)
    within_interval = (theta_test > ci_low) & (theta_test < ci_high)
    cicp = jnp.mean(within_interval, axis=0)
    
    widths = ci_high - ci_low
    mciw = jnp.mean(widths, axis=0)
    
    return cicp, mciw

def evaluation_models_metrics(
    flow_types, 
    models, 
    models_paths, 
    test_D, 
    test_theta_p
):

    results = {}

    for flow_name, template, path in zip(flow_types, models, models_paths):
        loaded_model = eqx.tree_deserialise_leaves(path, template)
        inference_model = eqx.nn.inference_mode(loaded_model)
        @eqx.filter_jit
        def run_inference(d, k):
            # Returns (num_samples, batch_size, dim) based on your provided code
            return inference_model.sample(d, k, n_samples)

        # 3. Warm-up (Compiles the JIT function for this specific batch size)
        warmup_key = jr.PRNGKey(0)
        _ = run_inference(test_D[:1], warmup_key).block_until_ready()

        # 4. Timing the Inference
        key = jr.PRNGKey(42)
        inf_key, _ = jr.split(key)
        start_time = time.perf_counter()
        
        # Run inference on the full test set
        # samples_raw shape: (n_samples, n_test, n_params)
        samples = run_inference(test_D, inf_key).block_until_ready()
        
        end_time = time.perf_counter()
        
        # Calculate time metrics
        total_time = end_time - start_time
        time_per_sample = total_time / test_size

        samples = denormalize_data(samples, mean_theta[None], std_theta[None])
        
        theta_approx_mean = jnp.mean(samples, axis=0)
        
        rmse = jnp.sqrt(jnp.mean((theta_approx_mean - test_theta_p) ** 2, axis=0))
        
        # R2 Score (calculated per parameter)
        r2 = np.array([
            r2_score(test_theta_p[:, j], theta_approx_mean[:, j]) 
            for j in range(test_theta_p.shape[1])
        ])
        
        cicp, mciw = cicp_mciw(samples, test_theta_p)

        item_name = os.path.normpath(path).split(os.sep)[-1][:-3]
        results[item_name] = {
            "rmse": rmse,
            "r2": r2,
            "cicp": cicp,
            "mciw": mciw,
            "inference_time": time_per_sample
        }
        
    return results

# plot in a spider chart
def plot_spider_from_results(metrics_list, target_dt0=0.05):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("colorblind", 5)
    data_target = {}
    for entry in metrics_list:
        # Check the first key to see if it contains our target noise string
        first_key = list(entry.keys())[0]
        if f"_{target_dt0}_1.0_force" in first_key:
            data_target = entry
            break
    if not data_target:
        print(f"No data found for target_dt0 {target_dt0}")
        return

    excluded_models = ['VAE', 'DDPM']
    filtered_models = {}
    for full_key, metrics in data_target.items():
        model_name = full_key.split('_')[0] # Extracts 'BNN', 'RealNVP', etc.
        if model_name not in excluded_models:
            filtered_models[model_name] = metrics

    categories = ['RMSE', '1-CICP', 'MCIW', 'Inference Time']
    num_vars = len(categories)

    # Extract and process values for remaining models
    processed_values = []
    model_names = list(filtered_models.keys())
    
    raw_values = []
    for name in model_names:
        m = filtered_models[name]
        # Calculate means across the 3 parameters
        avg_rmse = np.mean(m['rmse'])
        avg_one_minus_cicp = 1 - np.mean(m['cicp'])
        avg_mciw = np.mean(m['mciw'])
        # Use 'inference_time' (or 'time_per_sample' depending on your dict key)
        inf_time = m.get('inference_time', m.get('time_per_sample', 0))
        
        raw_values.append([avg_rmse, avg_one_minus_cicp, avg_mciw, inf_time])

    raw_values = np.array(raw_values)

    # 4. Normalization (0=Center/Best, 1=Outer/Worst)
    norm_min = raw_values.min(axis=0)
    norm_max = raw_values.max(axis=0)
    norm_values = (raw_values - norm_min) / (norm_max - norm_min)

    # 5. Radar Chart Plotting
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, name in enumerate(filtered_models):
        values = norm_values[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=3, label=name, marker='o')
        ax.fill(angles, values, color=colors[i], alpha=0.05)

    # Coordinate system setup
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels with padding
    # pad=30 pushes the text further away from the chart axes
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.tick_params(axis='x', pad=25) 

    # Fine-tune label alignments to prevent overlapping with the outer circle
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if np.allclose(angle, 0): # RMSE (Top)
            label.set_ha('center')
            label.set_va('bottom')
        elif np.allclose(angle, np.pi/2): # 1-CICP (Right)
            label.set_ha('left') # Moves it further right
            label.set_va('center')
        elif np.allclose(angle, np.pi): # MCIW (Bottom)
            label.set_ha('center')
            label.set_va('top')
        elif np.allclose(angle, 3*np.pi/2): # Inference Time (Left)
            label.set_ha('right') # Moves it further left
            label.set_va('center')

    # Hide the radial labels (0.2, 0.4...) for clarity
    ax.set_yticklabels([])

    # plt.title("Detailed Model Comparison\nCenter = Best Performance", 
    #           size=14, pad=40)

    # Position legend clearly outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'metrics_{target_dt0}_spider_chart.pdf'))

def plot_comprehensive_metrics(metrics_list):
    sns.set_theme(style="ticks")
    colors = [
        '#0072B2', # Deep Blue
        '#D55E00', # Vermillion (Orange-Red)
        '#009E73', # Bluish Green
        '#CC79A7', # Reddish Purple
        '#E69F00', # Orange
        '#56B4E9', # Sky Blue
        '#d0bbff'  # Purple
    ]

    dt0_levels = [0.5, 0.1, 0.05, 0.01]
    model_basenames = ['BNN', 'VAE', 'DDPM', 'SBM', 'FM', 'RealNVP', 'NSF']
    
    # Storage for plotting (Model -> List of values corresponding to noise_levels)
    results = {m: {'rmse': [], 'cicp': [], 'mciw': []} for m in model_basenames}

    fig, (ax_rmse, ax_uncert) = plt.subplots(1, 2, figsize=(24, 9))

    for dt in dt0_levels:
        target_dict = next((entry for entry in metrics_list if f"_{dt}_1.0_force" in list(entry.keys())[0]), {})
        for base in model_basenames:
            full_key = [k for k in target_dict.keys() if k.startswith(base)][0]
            m = target_dict[full_key]
            # Calculating means across the system parameters
            results[base]['rmse'].append(np.mean(m['rmse']))
            results[base]['cicp'].append(np.mean(m['cicp']))
            # Using mciw (or mciw depending on your dict key)
            results[base]['mciw'].append(np.mean(m.get('mciw', m.get('mciw', 0))))

    # --- Plot 1: Accuracy Curve (R2 Score) ---
    line_handles = []
    for i, model in enumerate(model_basenames):
        h, = ax_rmse.plot(dt0_levels, results[model]['rmse'], 
                     marker='o', markersize=8, linewidth=2, 
                     color=colors[i], label=model)
        line_handles.append(h)
    
    ax_rmse.set_xscale('log')
    ax_rmse.invert_xaxis()
    ax_rmse.set_xlabel(r'$\mathbf{log(dt_0)}$', fontsize=32, fontweight='bold')
    ax_rmse.set_ylabel('RMSE', fontsize=32, fontweight='bold')
    ax_rmse.grid(True, ls=":")
    ax_rmse.tick_params(axis='both', which='major', labelsize=28)

    # --- Plot 2: Uncertainty Quantification (Bar Charts) ---
    width = 0.11
    x = np.arange(len(dt0_levels))
    
    for i, model in enumerate(model_basenames):
        offset = (i - 3) * width
        ax_uncert.bar(x + offset, results[model]['mciw'], width, 
                      color=colors[i], edgecolor='white', label=f"{model} (Width)")

    ax_uncert.set_ylabel('MCIW', fontsize=32, fontweight='bold')
    ax_uncert.set_xticks(x)
    ax_uncert.set_xticklabels([f"${n}$" for n in dt0_levels])
    ax_uncert.set_xlabel(r'$\mathbf{dt_0}$', fontsize=32, fontweight='bold')
    ax_uncert.tick_params(axis='both', which='major', labelsize=28)

    # Secondary Y-axis: CICP (Curve as Reference)
    ax_cicp = ax_uncert.twinx()
    for i, model in enumerate(model_basenames):
        # We plot the line through the centers of the bar groups
        ax_cicp.plot(x, results[model]['cicp'], 
                     marker='D', markersize=5, linewidth=2, linestyle='--',
                     color=colors[i], label=f"{model} (Coverage)")

    ax_cicp.axhline(0.95, color='black', linestyle=':', linewidth=1.5)
    ax_cicp.set_ylabel('CICP', fontsize=32, fontweight='bold')
    ax_cicp.set_ylim(0.4, 1.05)
    ax_cicp.tick_params(axis='both', which='major', labelsize=28)

    sns.despine(ax=ax_rmse)
    sns.despine(ax=ax_uncert, right=False)
    fig.legend(handles=line_handles, labels=model_basenames, 
               loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=7, frameon=True, fontsize=32, title_fontsize=13)

    # Adjust layout to make room for the bottom legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(figure_dir, 'curve_chart_sample.pdf'))

def plot_noise_sensitivity(metrics_list):
    sns.set_theme(style="ticks")
    colors = sns.color_palette("Set2", 3)

    noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    model_basenames = ['RealNVP', 'NSF', 'FM']
    
    results = {m: {'rmse': [], 'cicp': [], 'mciw': []} for m in model_basenames}
    for n in noise_levels:
        if n == 0:
            target_dict = next((entry for entry in metrics_list if f"_0.05_1.0_force" in list(entry.keys())[0]), {})
        else:
            target_dict = next((entry for entry in metrics_list if f"_0.05_1.0_noise_{n}_force" in list(entry.keys())[0]), {})
        for base in model_basenames:
            full_key = [k for k in target_dict.keys() if k.startswith(base)][0]
            m = target_dict[full_key]
            # Calculating means across the system parameters
            results[base]['rmse'].append(np.mean(m['rmse']))
            results[base]['cicp'].append(np.mean(m['cicp']))
            # Using mciw (or mciw depending on your dict key)
            results[base]['mciw'].append(np.mean(m['mciw']))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    noise_levels_proxy = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for i, model in enumerate(model_basenames):
        ax[0].plot(noise_levels_proxy, results[model]['rmse'], 'o-', label=f'{model}', linewidth=2, markersize=7, color=colors[i])
        ax[1].plot(noise_levels_proxy, results[model]['mciw'], 'o-', label=f'{model}', linewidth=2, markersize=7, color=colors[i])
        ax[2].plot(noise_levels_proxy, results[model]['cicp'], 'o-', label=f'{model}', linewidth=2, markersize=7, color=colors[i])

    ax[0].set_xscale('log')  # Log scale for X-axis as requested
    ax[0].set_yscale('log')  # Log scale for Y-axis is usually better for RMSE growth
    ax[0].set_xticks(noise_levels_proxy, labels=[str(s) for s in noise_levels])
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_xlabel(r'Noise Level $\sigma$', fontsize=20)
    ax[0].set_title('Inference Error RMSE', fontsize=20)
    ax[0].legend(frameon=True, fontsize=20)
    ax[0].grid(True, which="both", ls="-", alpha=0.5)

    ax[1].set_xscale('log')  # Log scale for X-axis as requested
    ax[1].set_yscale('log')  
    ax[1].set_xticks(noise_levels_proxy, labels=[str(s) for s in noise_levels])
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    ax[1].set_xlabel(r'Noise Level $\sigma$', fontsize=20)
    ax[1].set_title('Uncertainty MCIW', fontsize=20)
    ax[1].grid(True, which="both", ls="-", alpha=0.5)

    ax[2].set_xscale('log')  # Log scale for X-axis as requested
    ax[2].set_xticks(noise_levels_proxy, labels=[str(s) for s in noise_levels])
    ax[2].tick_params(axis='both', which='major', labelsize=16)
    ax[2].set_xlabel(r'Noise Level $\sigma$', fontsize=20)
    ax[2].set_title('Coverage CICP', fontsize=20)
    ax[2].grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(figure_dir, 'stress_test_noise.pdf'))
    plt.show()


model_key = jr.PRNGKey(42)
feature_dim = 2 
hidden_dim_rnn = 64  # GRU hidden size
feature_latent_dim = 64
hidden_dim_time = 4
hidden_sizes = [64] * 2
param_dim = 3
param_names = [r'$\mathbf{m}$', r'$\mathbf{k}$', r'$\mathbf{c}$']
t1 = 1.0
test_key = jr.PRNGKey(0)
test_key, sample_key = jr.split(test_key, 2)
test_size = 200
n_samples = 100

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
results_dir = os.path.join(project_root, 'results')
figure_dir = os.path.join(project_root, 'evaluation')

# flow_types = ['RealNVP', 'FM', 'MALA']
# models = [create_model(flow_type) for flow_type in flow_types]
# noise_std_list = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# dt0 = 0.05
# metrics_noises = []
# for noise_std in noise_std_list:
#     if noise_std == 0:
#         data_set = f"mass_spring_damper_{dt0}_{t1}_force"
#     else:
#         data_set = f"mass_spring_damper_{dt0}_{t1}_noise_{noise_std}_force"
#     test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
#     test_D, mean, std = normalize_data(test_D)
#     test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)
#     model_paths = [os.path.join(results_dir, f"{flow_type}_{data_set}.pt") for flow_type in flow_types]
#     metrics_noises.append(evaluation_models_metrics(flow_types, models, model_paths, test_D, test_theta_p))
# plot_noise_sensitivity(metrics_noises)

flow_types = ['BNN', 'VAE', 'DDPM', 'SBM', 'FM', 'RealNVP', 'NSF']
models = [create_model(flow_type) for flow_type in flow_types]
dt0_list = [0.5, 0.1, 0.01, 0.05]
noise_std = 0
metrics_samples = []
for dt0 in dt0_list:
    data_set = f"mass_spring_damper_{dt0}_{t1}_force"
    test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
    test_D, mean, std = normalize_data(test_D)
    test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)
    model_paths = [os.path.join(results_dir, f"{flow_type}_{data_set}.pt") for flow_type in flow_types]
    metrics_samples.append(evaluation_models_metrics(flow_types, models, model_paths, test_D, test_theta_p))