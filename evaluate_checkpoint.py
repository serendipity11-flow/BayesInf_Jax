#!/usr/bin/env python3
"""
Comprehensive evaluation script for RealNVPFlowSDE checkpoint.
Extracts hyperparameters from checkpoint filename and evaluates model performance.
"""

import sys, os, re
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Setup paths
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    lorenz96_rhs,
)

# ============================================================================
# Checkpoint filename parser
# ============================================================================

def parse_checkpoint_name(filename):
    """
    Extract hyperparameters from checkpoint filename.
    Example: RealNVPFlowSDE_N12_h101_obs2_gmm3c_std0.5_dc128_dh256_dep3_dfh64_rnvp2_a1_z0provided_epsstep_ctx1_seg25_pwheun_custom_sn0.01_rw1_cw0.1_sched5to25.pt
    """
    params = {}
    
    # Extract model type
    if "RealNVP" in filename:
        params["flow_type"] = "realnvp"
    elif "NICE" in filename:
        params["flow_type"] = "nice"
    
    # Extract dimensions and hyperparameters with regex
    patterns = {
        "N": r"_N(\d+)_",
        "horizon": r"_h(\d+)_",
        "obs_interval": r"_obs(\d+)_",
        "obs_std": r"_std([\d.]+)_",
        "d_c": r"_dc(\d+)_",
        "d_hidden": r"_dh(\d+)_",
        "depth": r"_dep(\d+)_",
        "d_hidden_diffusion": r"_dfh(\d+)_",
        "n_realnvp_layers": r"_rnvp(\d+)_",
        "realnvp_alpha": r"_a([\d.]+)_",
        "observation_stride": r"_ctx(\d+)_",
        "seg_length": r"_seg(\d+)_",
        "shooting_noise_std": r"_sn([\d.]+)_",
        "rollout_loss_weight": r"_rw([\d.]+)_",
        "continuity_loss_weight": r"_cw([\d.]+)_",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            val = match.group(1)
            if key in ["N", "horizon", "obs_interval", "d_c", "d_hidden", "depth", 
                      "d_hidden_diffusion", "n_realnvp_layers", "observation_stride", "seg_length"]:
                params[key] = int(val)
            else:
                params[key] = float(val)
    
    # Check for GMM in filename
    gmm_match = re.search(r"_gmm(\d+)c_", filename)
    if gmm_match:
        params["obs_noise_type"] = "gmm"
        params["gmm_components"] = int(gmm_match.group(1))
    else:
        params["obs_noise_type"] = "gaussian"
    
    # Check for solver type
    if "pwheun" in filename:
        params["solver"] = "heun"
    elif "pwmilstein" in filename:
        params["solver"] = "milstein"
    elif "pwrk4" in filename:
        params["solver"] = "rk4"
    else:
        params["solver"] = "heun"  # default
    
    # Check for custom solver
    params["use_custom_solver"] = "custom" in filename
    
    # Check for multiple shooting
    params["use_multiple_shooting"] = params.get("seg_length", 0) > 0
    
    # Check for segment scheduling
    sched_match = re.search(r"_sched(\d+)to(\d+)", filename)
    if sched_match:
        params["seg_length_start"] = int(sched_match.group(1))
    else:
        params["seg_length_start"] = params.get("seg_length", 5)
    
    # Set default GMM parameters
    if params.get("obs_noise_type") == "gmm":
        params["gmm_weights"] = (0.70, 0.15, 0.15)
        params["gmm_offsets"] = (0.0, 2.5, -2.5)
        params["gmm_stds"] = (0.50, 1.50, 1.50)
    
    return params


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    ckpt_path = "/home/zhangyi/bayesinf_jax/results/results_state_estimation/RealNVPFlowSDE_N12_h101_obs2_gmm3c_std0.5_dc128_dh256_dep3_dfh64_rnvp2_a1_z0provided_epsstep_ctx1_seg25_pwheun_custom_sn0.01_rw1_cw0.1_sched5to25.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return
    
    # Parse checkpoint filename
    ckpt_name = os.path.basename(ckpt_path)
    params = parse_checkpoint_name(ckpt_name)
    
    print("=" * 70)
    print("EVALUATION: RealNVPFlowSDE on Lorenz-96")
    print("=" * 70)
    print(f"\nCheckpoint: {ckpt_name}")
    print(f"\nExtracted hyperparameters:")
    for key, val in sorted(params.items()):
        print(f"  {key:30s} = {val}")
    
    # Set evaluation parameters
    n_eval = 100
    n_particles = 200
    n_plot = 6
    ci = 90.0
    seed = 0
    
    # Get dimensions
    d_x = params.get("N", 12)
    d_y = d_x // params.get("obs_interval", 2)
    
    # Create physics function
    print(f"\nGenerating test data (N={d_x}, horizon={params.get('horizon', 100)})...")
    k_data = jr.PRNGKey(seed)
    
    data = generate_lorenz96_dataset(
        k_data,
        n_samples=2000,
        horizon=params.get("horizon", 100),
        N=d_x,
        obs_interval=params.get("obs_interval", 2),
        obs_std=params.get("obs_std", 0.5),
        obs_noise_type=params.get("obs_noise_type", "gmm"),
        gmm_weights=params.get("gmm_weights") if params.get("obs_noise_type") == "gmm" else None,
        gmm_offsets=params.get("gmm_offsets") if params.get("obs_noise_type") == "gmm" else None,
        gmm_stds=params.get("gmm_stds") if params.get("obs_noise_type") == "gmm" else None,
        dt=0.02,
        F=8.0,
        spinup_steps=500,
        d_u=1,
    )
    
    _, _, test_data = train_val_test_split(data, jr.fold_in(k_data, 1), 
                                          train_frac=0.8, val_frac=0.1)
    
    x_test_raw = np.array(test_data["x"])
    x0_test_raw = np.array(test_data["x0"])
    y_test_raw = np.array(test_data["y"])
    M_test = x_test_raw.shape[0]
    n_eval = min(n_eval, M_test)
    T = params.get("horizon", 100)
    
    obs_idx = np.arange(0, d_x, params.get("obs_interval", 2))
    unobs_idx = np.array([i for i in range(d_x) if i not in obs_idx])
    
    # Load normalization stats
    norm_stats_path = ckpt_path.replace(".pt", "_norm_stats.npz")
    if not os.path.exists(norm_stats_path):
        print(f"WARNING: Norm stats not found: {norm_stats_path}")
        print("         Using identity normalization")
        x_mean, x_std = np.zeros(d_x), np.ones(d_x)
        y_mean, y_std = np.zeros(d_y), np.ones(d_y)
    else:
        ns = np.load(norm_stats_path)
        x_mean = ns["x_mean"]
        x_std = ns["x_std"]
        y_mean = ns["y_mean"]
        y_std = ns["y_std"]
        print(f"Normalization stats loaded: x mean≈{x_mean.mean():.2f} std≈{x_std.mean():.2f}")
    
    # Normalize test data
    x_test = jnp.array((x_test_raw - x_mean) / x_std)
    x0_test = jnp.array((x0_test_raw - x_mean) / x_std)
    y_test = jnp.array((y_test_raw - y_mean) / y_std)
    
    # Build and load model
    print(f"\nLoading checkpoint...")
    _F = 8.0
    _xm = jnp.array(x_mean)
    _xs = jnp.array(x_std)
    physics_fn = lambda z: lorenz96_rhs(_xm + _xs * z, _F) / _xs
    
    model = RealNVPFlowSDE(
        d_x=d_x,
        d_y=d_y,
        d_c=params.get("d_c", 64),
        d_hidden_enc=params.get("d_c", 64),  # Use d_c as default for d_hidden_enc
        d_hidden=params.get("d_hidden", 128),
        d_hidden_diffusion=params.get("d_hidden_diffusion", 64),
        depth=params.get("depth", 3),
        time_emb_dim=32,
        n_realnvp_layers=params.get("n_realnvp_layers", 2),
        realnvp_alpha=params.get("realnvp_alpha", 1.0),
        flow_type=params.get("flow_type", "realnvp"),
        n_nice_layers=2,
        nice_noise_scale=0.5,
        observation_stride=params.get("observation_stride", 1),
        time_step=0.02,
        key=jr.PRNGKey(42),
        physics_fn=physics_fn,
    )
    
    try:
        model = eqx.tree_deserialise_leaves(ckpt_path, model)
        model = eqx.nn.inference_mode(model)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # Posterior sampling
    print(f"\nSampling posterior ({n_eval} sequences, {n_particles} particles)...")
    
    @eqx.filter_jit
    def _sample_chunk(model, y_chunk, x0_chunk, key, n_particles, solver_name):
        B = y_chunk.shape[0]
        keys = jr.split(key, B)
        return jax.vmap(
            lambda yi, x0i, ki: model.sample_posterior_custom_solver(
                yi, x0i, ki, n_particles, solver_name,
                diffusion_temp=1.0,
                infer_initial=False,
                observation_stride=params.get("observation_stride", 1),
            )
        )(y_chunk, x0_chunk, keys)
    
    chunk = 20
    k_eval = jr.PRNGKey(999)
    all_samps = []
    
    for start in range(0, n_eval, chunk):
        end = min(start + chunk, n_eval)
        yb = y_test[start:end]
        k_c = jr.fold_in(k_eval, start)
        try:
            s = np.array(_sample_chunk(
                model, yb, x0_test[start:end], k_c, n_particles, 
                params.get("solver", "heun")
            ))
            all_samps.append(s)
            print(f"  Sampled [{start:3d}:{end:3d}]")
        except Exception as e:
            print(f"  ERROR sampling [{start:3d}:{end:3d}]: {e}")
            return
    
    all_samps = np.concatenate(all_samps, axis=0)
    all_samps = all_samps * x_std[None, None, None] + x_mean[None, None, None]
    
    # Compute metrics
    print(f"\nComputing metrics...")
    post_mean = all_samps.mean(axis=1)
    lo = np.percentile(all_samps, (100 - ci) / 2, axis=1)
    hi = np.percentile(all_samps, 100 - (100 - ci) / 2, axis=1)
    
    x_true = x_test_raw[:n_eval]
    
    rmse_all = float(np.mean(np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(1, 2)))))
    rmse_obs = float(np.sqrt(np.mean((x_true[:, :, obs_idx] - post_mean[:, :, obs_idx]) ** 2)))
    rmse_unobs = float(np.sqrt(np.mean((x_true[:, :, unobs_idx] - post_mean[:, :, unobs_idx]) ** 2))) if len(unobs_idx) else float("nan")
    mae_all = float(np.mean(np.abs(x_true - post_mean)))
    
    cov_per_dim = np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1))
    coverage = float(cov_per_dim.mean())
    
    # Calibration: expected coverage vs actual for different CI levels
    ci_levels = [50, 68, 90, 95, 99]
    calibration = {}
    for ci_level in ci_levels:
        lo_cal = np.percentile(all_samps, (100 - ci_level) / 2, axis=1)
        hi_cal = np.percentile(all_samps, 100 - (100 - ci_level) / 2, axis=1)
        cov_cal = float(np.mean((x_true >= lo_cal) & (x_true <= hi_cal)))
        calibration[ci_level] = cov_cal
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nPosterior Mean Accuracy:")
    print(f"  RMSE (all dims):          {rmse_all:.4f}")
    print(f"  RMSE (observed dims):     {rmse_obs:.4f}")
    print(f"  RMSE (unobserved dims):   {rmse_unobs:.4f}")
    print(f"  MAE:                      {mae_all:.4f}")
    
    print(f"\nUncertainty Quantification @ {ci:.0f}% CI:")
    print(f"  Coverage:                 {coverage * 100:.1f}%")
    print(f"  Coverage per dimension:   min={cov_per_dim.min()*100:.1f}%  "
          f"max={cov_per_dim.max()*100:.1f}%  mean={cov_per_dim.mean()*100:.1f}%")
    
    print(f"\nCalibration (expected vs actual coverage):")
    for ci_level in ci_levels:
        expected = ci_level / 100.0
        actual = calibration[ci_level]
        error = abs(actual - expected) * 100
        print(f"  {ci_level:2d}%:  expected={expected:.3f}  actual={actual:.3f}  error={error:.1f}pp")
    
    print(f"\nEvaluation completed for {n_eval} test sequences")
    print("=" * 70)
    
    # Optionally save metrics
    results_dict = {
        "checkpoint": ckpt_name,
        "n_eval": n_eval,
        "n_particles": n_particles,
        "rmse_all": rmse_all,
        "rmse_obs": rmse_obs,
        "rmse_unobs": rmse_unobs,
        "mae": mae_all,
        "coverage": coverage,
        "calibration": calibration,
        "coverage_per_dim": cov_per_dim,
    }
    
    results_path = ckpt_path.replace(".pt", "_eval_results.npz")
    np.savez(results_path, **results_dict)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
