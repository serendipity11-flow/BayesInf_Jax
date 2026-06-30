import sys
import os
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import seaborn as sns 
import jax.random as jr
import equinox as eqx
import jax
from functools import partial
import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import os
from bayesinf.utils.generate_data import normalize_data, denormalize_data

multi_modal = True

if multi_modal:
    from bayesinf.utils.generate_data import (generate_mass_spring_data2, mass_spring_ode2)
    generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(generate_mass_spring_data2)

else:
    from bayesinf.utils.generate_data import (generate_mass_spring_data_l, mass_spring_ode_l)
    generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(generate_mass_spring_data_l)

def generate_comparison_grid_mm(model_names, models, model_paths, test_D, test_theta_p, indices=[0, 49, 99, 199]):
    n_rows_data = len(indices) # 4
    n_cols = len(model_paths)
    idx_k1, idx_k3 = 0, 1 
    
    # Mapping for 5-row grid (skipping index 2 for the gap)
    grid_rows = [0, 1, 3, 4] 

    # --- Step 1: Data Collection (Unchanged) ---
    plot_data_cache = [[None for _ in range(n_cols)] for _ in range(n_rows_data)]
    for col_idx, (name, template, path) in enumerate(zip(model_names, models, model_paths)):
        loaded_model = eqx.tree_deserialise_leaves(path, template)
        inference_model = eqx.nn.inference_mode(loaded_model)
        samples = inference_model.sample(test_D, sample_key, n_samples)
        samples = denormalize_data(samples, mean_theta[None], std_theta[None])
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

    plt.savefig(os.path.join(figure_dir, "model_comparison_grid_multi_modal.pdf"), dpi=200, bbox_inches='tight')
    plt.show()

def generate_correlation_comparison_grid(model_names, models, model_paths, test_D, test_theta_p, indices=[0, 199]):
    n_cols = len(model_paths)
    n_rows_data = 4
    grid_rows = [0, 1, 3, 4] # Gap at index 2
    
    row_configs = []
    for idx in indices:
        row_configs.append((idx, 0, 1)) # m vs k
        row_configs.append((idx, 0, 2)) # m vs c

    # Step 1: Data Collection (Unchanged)
    row_data_cache = [[] for _ in range(n_rows_data)]
    for col_idx, (name, template, path) in enumerate(zip(model_names, models, model_paths)):
        loaded_model = eqx.tree_deserialise_leaves(path, template)
        inference_model = eqx.nn.inference_mode(loaded_model)
        samples = inference_model.sample(test_D, sample_key, n_samples)
        samples = denormalize_data(samples, mean_theta[None], std_theta[None])
        for row_idx, (s_id, ix, iy) in enumerate(row_configs):
            row_data_cache[row_idx].append((samples[:, s_id, ix], samples[:, s_id, iy]))

    # Step 2: Plotting with GridSpec
    fig = plt.figure(figsize=(n_cols * 5.5, (n_rows_data * 5) + 0.5))
    gs = fig.add_gridspec(5, n_cols, height_ratios=[1, 1, 0.1, 1, 1])
    axes_list = []

    for r_idx_data, (sample_id, idx_x, idx_y) in enumerate(row_configs):
        gt_x, gt_y = test_theta_p[sample_id, idx_x], test_theta_p[sample_id, idx_y]
        
        # Row-wise normalization
        max_val = 1
        for c in range(n_cols):
            ex, ey = row_data_cache[r_idx_data][c]
            counts, _, _ = np.histogram2d(ex, ey, bins=40)
            max_val = max(max_val, counts.max())
        row_norm = mcolors.LogNorm(vmin=1, vmax=max_val)

        row_axes = []
        for col_idx, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[grid_rows[r_idx_data], col_idx])
            row_axes.append(ax)
            est_x, est_y = row_data_cache[r_idx_data][col_idx]

            sns.kdeplot(x=est_x, y=est_y, ax=ax, cmap="Blues", fill=True, thresh=0.05, levels=8, alpha=0.6)
            hb = ax.hexbin(est_x, est_y, gridsize=40, cmap='coolwarm', mincnt=2, norm=row_norm, alpha=0.9)
            if col_idx == n_cols - 1:
                cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                cb.set_ticks([]) 
                cb.ax.yaxis.set_major_locator(NullLocator()) # Kill the major logic
                cb.ax.yaxis.set_minor_locator(NullLocator()) # Kill the minor logic (small ticks)
                cb.ax.tick_params(axis='y', which='both', length=0)
                cb.set_label("Counts", size=30, labelpad=15)

            ax.scatter(est_x, est_y, alpha=0.15, s=15, color='purple')
            ratio = gt_y / gt_x
            x_line = np.linspace(min(est_x)*0.95, max(est_x)*1.05, 100)
            line_handle, = ax.plot(x_line, ratio * x_line, 'k--', linewidth=3, 
                                   label=f'{param_names[idx_y]}/{param_names[idx_x]} = '+rf'$\mathbf{{{ratio:.2f}}}$')
            ax.scatter([gt_x], [gt_y], c='gold', s=450, marker='*', edgecolors='k', zorder=10)

            ax.legend(handles=[line_handle], fontsize=30, loc='upper left', 
                      framealpha=0.6, handlelength=1.2, handletextpad=0.4)
            if r_idx_data == n_rows_data - 1: 
                ax.set_xlabel(f"{param_names[idx_x]}", fontsize=32)
            ax.set_xticks([]); ax.set_yticks([])
            if r_idx_data == 0: 
                ax.set_title(model_name, fontsize=32, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{param_names[idx_y]}", fontsize=32)
        axes_list.append(row_axes)

    # Step 3: Global Legend & Line
    global_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=15, label='Posterior'),
                      Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='k', markersize=25, label='Ground Truth')]
    fig.legend(handles=global_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=32, frameon=True, shadow=True)

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.97])
    fig.canvas.draw() 

    # Line placement
    pos_top = axes_list[1][0].get_position()
    pos_bot = axes_list[2][0].get_position()
    y_mid = (pos_top.y0 + pos_bot.y1) / 2

    separator = Line2D([0.05, 0.98], [y_mid, y_mid], transform=fig.transFigure, color='blue', linewidth=4, linestyle='--', alpha=0.8)
    fig.add_artist(separator)
    plt.savefig(os.path.join(figure_dir, "model_comparison_grid_linear.pdf"), dpi=200, bbox_inches='tight')
    plt.show()

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
    else:
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, 
                            hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                            encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                            rnn_type='lstm', flow_type=flow_type)  
    return model     

model_key = jr.PRNGKey(42)
feature_dim = 2 
hidden_dim_rnn = 64  # GRU hidden size
feature_latent_dim = 64
hidden_dim_time = 4
hidden_sizes = [64] * 2
param_dim = 2 if multi_modal else 3
param_names = [r'$\mathbf{k_1}$', r'$\mathbf{k_3}$'] if multi_modal else [r'$\mathbf{m}$', r'$\mathbf{k}$', r'$\mathbf{c}$']
dt0 = 0.05
t1 = 1.0
noise_std = 0.0

test_key = jr.PRNGKey(0)
test_key, sample_key = jr.split(test_key, 2)
test_size = 200
n_samples = 100
test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
test_D, mean, std = normalize_data(test_D)
test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)
flow_types = ['BNN', 'VAE', 'DDPM', 'SBM', 'FM', 'RealNVP', 'NSF']
models = [create_model(flow_type) for flow_type in flow_types]

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
results_dir = os.path.join(project_root, 'results', 'results_parameter_identification')
figure_dir = os.path.join(project_root, 'evaluation')

if multi_modal:
    data_set = "mass_spring_damper_0.05_1.0_multi_modal"
else:
    data_set = "mass_spring_damper_0.05_1.0_linear"
model_paths = [os.path.join(results_dir, f"{flow_type}_{data_set}.pt") for flow_type in flow_types]
if multi_modal:
    generate_comparison_grid_mm(flow_types, models, model_paths, test_D, test_theta_p)
else: 
    generate_correlation_comparison_grid(flow_types, models, model_paths, test_D, test_theta_p)
