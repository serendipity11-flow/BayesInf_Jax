import sys
import os
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
from jax._src.source_info_util import current
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns 
import jax.random as jr
import equinox as eqx
import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import os
from sklearn.metrics import r2_score
from bayesinf.utils.generate_data import normalize_data, denormalize_data


def confidence_interval(samples, lower=0.025, upper=0.975):
    ci_low = jnp.quantile(samples, lower, axis=0)
    ci_high = jnp.quantile(samples, upper, axis=0)
    return ci_low, ci_high


def cicp_mciw(theta_approx, theta_test):
    ci_low, ci_high = confidence_interval(theta_approx)
    within_interval = (theta_test > ci_low) & (theta_test < ci_high)
    cicp = jnp.mean(within_interval, axis=0)
    mciw = jnp.mean(ci_high - ci_low, axis=0)
    return cicp, mciw

misspecification = False
ood = True

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
                        rnn_type='lstm', flow_type='realnvp', metr_steps_per_block=2,
                        lang_steps=1, stepsize=5e-3)          
    else:
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, 
                            hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                            encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                            rnn_type='lstm', flow_type=flow_type)  
    return model     

def plot_distribution_grid(sample_results, data_dict, idx=0, x_p=0, y_p=2):
    """
    sample_results: A nested dict [model_name][scale][mode] = array of posterior samples
    true_val: The actual parameter value used to generate the test data
    """

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)

    for i, flow in enumerate(flow_types):
        for j, scale in enumerate(scales):
            ax = axes[i, j]

            for k, mode in enumerate(modes):
                x = sample_results[scale][flow][mode][:, idx, x_p]
                y = sample_results[scale][flow][mode][:, idx, y_p]

                # Plot the KDE (Posterior Distribution)
                sns.kdeplot(x=x, y=y, ax=ax, 
                            fill=True, alpha=0.5, 
                            color=colors[k], 
                            levels=5, # Number of contour rings
                            thresh=0.05)
                sns.kdeplot(x=x, y=y, ax=ax, label=mode,
                            color=colors[k], linewidths=1.5,
                            levels=[0.1], alpha=0.8)

            # Draw the Ground Truth vertical line
            gx, gy = data_dict[scale]['test_theta_p'][idx, x_p], data_dict[scale]['test_theta_p'][idx, y_p]
            ax.scatter(gx, gy, 
                    marker='+', s=200, linewidth=2, color='red', label='Ground Truth')
            ax.axhline(gy, color='black', alpha=0.1, linestyle='--')
            ax.axvline(gx, color='black', alpha=0.1, linestyle='--')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if i == 0: 
                ax.set_title(f"Level: {scale}", fontsize=20, fontweight='bold')
            if j == 0: 
                ax.set_ylabel(f"{flow}\n{param_names[y_p]}", fontsize=20, fontweight='bold')
            if i == 2: 
                ax.set_xlabel(f"{param_names[x_p]}", fontsize=20, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='+', color='red', label='Ground Truth', markersize=20, linestyle='None'),
        mpatches.Patch(color=colors[0], label='Original', alpha=0.5),
        mpatches.Patch(color=colors[1], label='retrain_cn', alpha=0.5),
        mpatches.Patch(color=colors[2], label='retrain', alpha=0.5)
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=4, frameon=False, fontsize=20)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(figure_dir, 'misspecification_distribution.pdf' if misspecification else 'ood_distribution.pdf'))
    plt.show()

def plot_global_residuals(sample_results, data_dict, idx_x=0, idx_y=2, n_sub=100):
    """
    Plots a 3x3 grid showing the global distribution of residuals (Error) 
    across all 200 test scenarios.
    """
    sns.set_context("paper", font_scale=1.8)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)

    for i, flow in enumerate(flow_types):
        for j, scale in enumerate(scales):
            ax = axes[i, j]
            ax.axhline(0, color='black', alpha=0.3, linestyle='--')
            ax.axvline(0, color='black', alpha=0.3, linestyle='--')
            ax.scatter(0, 0, marker='+', s=300, color='red', zorder=10)

            for k, mode in enumerate(modes):
                samples = sample_results[scale][flow][mode] 
                truths = data_dict[scale]['test_theta_p']   
                
                # Calculate residuals
                res = samples - truths[np.newaxis, :, :]
                res_x = res[:, :, idx_x].flatten()
                res_y = res[:, :, idx_y].flatten()

                # --- SPEED OPTIMIZATION: Subsampling ---
                if len(res_x) > n_sub:
                    idx = np.random.choice(len(res_x), n_sub, replace=False)
                    res_x, res_y = res_x[idx], res_y[idx]

                # Plot KDE - Use fewer levels and a higher threshold for speed
                sns.kdeplot(x=res_x, y=res_y, ax=ax, fill=True, alpha=0.6, 
                            color=colors[k], levels=3, thresh=0.1)

            if i == 0: ax.set_title(f"Level: {scale}", fontsize=20, fontweight='bold')
            if j == 0: ax.set_ylabel(f"{flow}\n$\\Delta$ {param_names[idx_y]}", fontsize=20, fontweight='bold')
            if i == 2: ax.set_xlabel(fr"$\Delta$ {param_names[idx_x]}", fontsize=20, fontweight='bold')

            ax.grid(axis='y', alpha=0.3)
            TICK_SIZE = 20
            ax.tick_params(axis='y', labelsize=TICK_SIZE)
            ax.tick_params(axis='x', labelsize=TICK_SIZE)

    # Global Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='+', color='red', label='Ground Truth', markersize=20, linestyle='None'),
        mpatches.Patch(color=colors[0], label='Original', alpha=0.5),
        mpatches.Patch(color=colors[1], label='retrain_cn', alpha=0.5),
        mpatches.Patch(color=colors[2], label='retrain', alpha=0.5)
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=4, frameon=False, fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(figure_dir, 'ood_residual.pdf' if misspecification else 'ood_residual.pdf'))
    plt.show()

def plot_metrics_summary(metrics_results):
    """
    results: dict[model][mode][scale] = {'rmse': [], 'mciw': [], 'cicp': []}
    Each list contains values for the 200 samples.
    """
    
    # Professional "Muted" Palette
    # Red/Orange for Failure, Blue for Partial Fix, Green for Full Fix

    flow_hatches = {'RealNVP': '', 'NSF': '///', 'FM': '...'}
    metric_keys = ['rmse', 'mciw', 'cicp']
    titles = ['Inference Error (RMSE)', 'Uncertainty Width (MCIW)', 'Coverage (CICP)']

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    # Bar width settings
    width = 0.08  # Width of a single bar
    group_gap = 0.05 # Gap between models (RealNVP, NSF, FM)
    level_gap = 0.4 # Gap between levels (0.5, 1.0, 2.0)

    for i, m_key in enumerate(metric_keys):
        ax = axes[i]
        
        for s_idx, s in enumerate(scales):
            # Calculate base position for this Misspecification Level
            base_pos = s_idx * (3 * 3 * width + 2 * group_gap + level_gap)
            
            for f_idx, flow in enumerate(flow_types):
                # Calculate position for this specific model group
                model_pos = base_pos + f_idx * (3 * width + group_gap)
                
                for m_idx, mode in enumerate(modes):
                    # Final bar position
                    bar_pos = model_pos + m_idx * width
                    
                    val = metrics_results[s][flow][mode][m_key]
                    y_val = np.mean(val) if isinstance(val, (list, np.ndarray)) else val
                    y_err = (np.std(val) / np.sqrt(len(val))) if isinstance(val, (list, np.ndarray)) else 0
                    
                    ax.bar(bar_pos, y_val, width, 
                           yerr=y_err, capsize=2,
                           color=colors[m_idx], 
                           hatch=flow_hatches[flow],
                           edgecolor='white', linewidth=0.5)

        # Formatting
        ax.set_yscale('log') # Log scale for ALL metrics as requested
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, which='both')
        ax.set_ylabel(titles[i], fontsize=16, fontweight='bold')
        ax.set_xticks([])

        if m_key == 'cicp':
            ax.axhline(0.95, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        tick_pos = []
        for s_idx in range(len(scales)):
            center = s_idx * (3 * 3 * width + 2 * group_gap + level_gap) + (4.5 * width + group_gap)
            tick_pos.append(center)
        
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"Level {s}" for s in scales], fontsize=15)
    # Create a complex legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], label='Original (Stressed)'),
        mpatches.Patch(facecolor='gray', hatch='', label='RealNVP', edgecolor='white'),
        mpatches.Patch(facecolor=colors[1], label='Retrain CN Only'),
        mpatches.Patch(facecolor='gray', hatch='///', label='NSF', edgecolor='white'),
        mpatches.Patch(facecolor=colors[2], label='Full Retrain'),
        mpatches.Patch(facecolor='gray', hatch='...', label='FM', edgecolor='white')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=3, frameon=False, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(os.path.join(figure_dir, 'misspecification_metrics.pdf' if misspecification else 'ood_metrics.pdf'))
    plt.show()

model_key = jr.PRNGKey(42)
feature_dim = 2 
hidden_dim_rnn = 64  # GRU hidden size
feature_latent_dim = 64
hidden_dim_time = 4
hidden_sizes = [64] * 2
param_dim = 3
param_names = [r'$\mathbf{m}$', r'$\mathbf{k}$', r'$\mathbf{c}$']
dt0 = 0.05
t1 = 1.0
noise_std = 0.0

test_key = jr.PRNGKey(0)
test_key, sample_key = jr.split(test_key, 2)
test_size = 200
n_samples = 100
data_set = f"mass_spring_damper_{dt0}_{t1}_force"

test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
test_D, mean, std = normalize_data(test_D)
test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
results_dir = os.path.join(project_root, 'results', 'results_parameter_identification')
figure_dir = os.path.join(project_root, 'evaluation')

sample_results = {}
data_dict = {}
metrics_results = {}
flow_types = ['RealNVP', 'NSF', 'FM']
scales = ['0.5', '1.0', '2.0']
modes = ['Original', 'retrain_cn', 'retrain']
colors = ['#E76F51', '#2A9D8F', '#836bc2']

for s, _scale in enumerate(scales):
    sample_results[_scale] = {}
    data_dict[_scale] = {}
    metrics_results[_scale] = {}
    if misspecification:
        _generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std', 'beta', 'alpha'])(
                             generate_mass_spring_data_f)
        test_D, test_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, beta=1*float(_scale), alpha=1*float(_scale))
        test_D = normalize_data(test_D, mean, std)
        data_dict[_scale]['test_D_normalized'] = test_D
        data_dict[_scale]['test_theta_p'] = test_theta_p
    elif ood:
        _generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std', 'shift'])(
                                         generate_mass_spring_data_f)    
        test_D, test_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, shift=(0.1*float(_scale), 1*float(_scale), 0.5*float(_scale)))
        test_D = normalize_data(test_D, mean, std)
        data_dict[_scale]['test_D_normalized'] = test_D
        data_dict[_scale]['test_theta_p'] = test_theta_p
    else:
        test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
        test_D, mean, std = normalize_data(test_D)
        test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)
        data_dict[_scale]['test_D_normalized'] = test_D
        data_dict[_scale]['test_theta_p'] = test_theta_p
    for m, _flow in enumerate(flow_types):
        sample_results[_scale][_flow] = {}
        metrics_results[_scale][_flow] = {}
        for r, _mode in enumerate(modes):
            if _mode == 'Original':
                suffix = ""
            else:
                if misspecification:
                    suffix = f"_misspecification_{_scale}_{_mode}"
                elif ood:
                    suffix = f"_ood_{_scale}_{_mode}"
                else:
                    suffix = f"_{_mode}"

            model_path = os.path.join(results_dir, f"{_flow}_{data_set}{suffix}.pt")
            model = create_model(_flow)
            inference_model = eqx.tree_deserialise_leaves(model_path, model)
            inference_model = eqx.nn.inference_mode(inference_model)
            @eqx.filter_jit
            def run_inference(d, k):
                # Returns (num_samples, batch_size, dim) based on your provided code
                return inference_model.sample(d, k, n_samples)
            
            inf_key, _ = jr.split(sample_key)
            samples = run_inference(test_D, inf_key).block_until_ready()
            samples = denormalize_data(samples, mean_theta[None], std_theta[None])
            sample_results[_scale][_flow][_mode] = samples
            metrics_results[_scale][_flow][_mode] = {}
            theta_approx_mean = jnp.mean(samples, axis=0)
            # metrics
            rmse = jnp.sqrt(jnp.mean((theta_approx_mean - test_theta_p) ** 2, axis=0))
            metrics_results[_scale][_flow][_mode]['rmse'] = rmse
            # R2 Score (calculated per parameter)
            r2 = np.array([
                r2_score(test_theta_p[:, j], theta_approx_mean[:, j]) 
                for j in range(test_theta_p.shape[1])
            ])
            metrics_results[_scale][_flow][_mode]['r2'] = r2
            cicp, mciw = cicp_mciw(samples, test_theta_p)
            metrics_results[_scale][_flow][_mode]['cicp'] = cicp
            metrics_results[_scale][_flow][_mode]['mciw'] = mciw

if misspecification:
    plot_distribution_grid(sample_results, data_dict)
elif ood:
    plot_global_residuals(sample_results, data_dict)
