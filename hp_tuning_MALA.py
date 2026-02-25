import sys
import os
import wandb
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
import optax
from functools import partial
from tqdm import tqdm

# --- Path Setup (Same as your original script) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# --- Imports from your library ---
from bayesinf.utils.generate_data import normalize_data, denormalize_data
from bayesinf.utils.generate_data import (generate_mass_spring_data_f, mass_spring_ode_f)
from bayesinf.models.MCMC_Flow import CondSNF, make_step

# --- 1. Define the Sweep Configuration ---
sweep_configuration = {
    'method': 'grid',  # Using grid to try every combination exactly once
    'name': 'MALA_Tuning_MassSpring',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss'
    },
    'parameters': {
        # The specific hyperparameters you requested
        'n_stochastic': {'values': [1, 2, 4]},
        'metr_steps_per_block': {'values': [1, 2, 3]},
        'lang_steps': {'values': [1, 2]},
        'stepsize': {'values': [5e-1, 5e-3, 1e-2]},
        
        # Fixed training params (can be turned into ranges if needed)
        'learning_rate': {'value': 0.01},
        'batch_size': {'value': 256}
    }
}

# --- 2. Define the Training Function ---
def train_sweep():
    with wandb.init() as run:
        config = wandb.config
        
        # --- Fixed Constants (Matches defaults from your original argparse) ---
        dt0 = 0.05
        t1 = 1.0
        noise_std = 0.0
        feature_dim = 2
        hidden_dim_rnn = 64
        feature_latent_dim = 64
        hidden_sizes = [64] * 2
        param_dim = 3  # [m, k, c]
        
        # Prepare Keys
        key = jr.PRNGKey(5)
        model_key, train_key, data_key = jr.split(key, 3)

        # --- Data Generation Setup ---
        # JIT the generator
        generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
            generate_mass_spring_data_f
        )

        # Generate Test/Normalization Data
        test_size = 200
        # Note: We generate a small batch just to calculate mean/std for normalization
        ref_D, ref_theta_p = generate_data_jit(jr.PRNGKey(0), dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
        _, mean, std = normalize_data(ref_D)
        _, mean_theta, std_theta = normalize_data(ref_theta_p)

        # --- Define Dynamics & Log Posterior (Specifically for MALA) ---
        def dynamics_fn(t, state_norm, theta_norm):
            state = denormalize_data(state_norm, jnp.squeeze(mean), jnp.squeeze(std))
            theta = denormalize_data(theta_norm, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
            deriv = jax.vmap(lambda _t, _s: mass_spring_ode_f(_t, _s, (theta,)))(t, state)
            deriv_norm = normalize_data(deriv, jnp.squeeze(mean), jnp.squeeze(std))
            return deriv_norm

        def log_posterior_fn(x, y):
            num_steps = int(t1/dt0)
            ts = jnp.linspace(0, t1-dt0, num_steps)
            
            # Handling dimensions based on your original logic
            if x.ndim > 1:
                states = y[:, 1:] - y[:, :-1]
            else:
                states = y[1:] - y[:-1]

            states_deriv = states / dt0
            if x.ndim > 1:
                pred_deriv = jax.vmap(dynamics_fn, in_axes=(None, 0, 0))(ts, states, x)
            else:
                pred_deriv = dynamics_fn(ts, states, x)
            
            residual = pred_deriv - states_deriv
            std_val = max(0.01, noise_std)
            
            if x.ndim > 1:
                log_likelihood = -0.5 * jnp.sum(residual ** 2, axis=(1, 2)) / (num_steps * std_val ** 2)
                log_prior = -0.5 * jnp.sum(x**2, axis=-1)
            else:
                log_likelihood = -0.5 * jnp.sum(residual ** 2) / (num_steps * std_val ** 2)
                log_prior = -0.5 * jnp.sum(x ** 2)
            return log_likelihood + log_prior

        # --- Initialize Model using Sweep Config ---
        # This is where the magic happens: we inject the sweep parameters
        model = CondSNF(
            param_dim, feature_dim, log_posterior_fn, key=model_key, n_coupling=8,
            hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
            encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn,
            rnn_type='lstm', flow_type='realnvp',
            
            # HYPERPARAMETERS FROM WANDB
            n_stochastic=config.n_stochastic,
            metr_steps_per_block=config.metr_steps_per_block,
            lang_steps=config.lang_steps,
            stepsize=config.stepsize
        )

        # --- Optimization Setup ---
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), 
                                optax.adamw(learning_rate=0.01, 
                                weight_decay=1e-4))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        
        # Shorter training for tuning (e.g., 200 epochs), or use full 10000 if resources allow
        num_epochs = 500  
        batch_size = config.batch_size
        patience = 50
        best_val_loss = np.inf
        patience_counter = 0

        # --- Training Loop ---
        print(f"--> Starting Run: n_stoch={config.n_stochastic}, steps={config.metr_steps_per_block}, lang={config.lang_steps}, lr={config.stepsize}")
        
        for i in range(num_epochs):
            # Generate new data on the fly (infinite dataset approach from original script)
            data_key, train_key = jr.split(train_key)
            batch_D, batch_theta_p = generate_data_jit(data_key, dt0=dt0, t1=t1, n_samples=batch_size, noise_std=noise_std)
            
            # Normalize
            batch_D = normalize_data(batch_D, mean, std)
            batch_theta_p = normalize_data(batch_theta_p, mean_theta, std_theta)
            
            # Step
            model, opt_state, loss = make_step(model, batch_D, batch_theta_p, train_key, opt_state, optimizer)
            
            # Logging
            wandb.log({'epoch': i, 'train_loss': loss})

            # Checkpoint / Patience
            if loss < best_val_loss:
                best_val_loss = loss
                # Update summary metric for the sweep to track
                wandb.run.summary["val_loss"] = loss 
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {i}")
                break

# --- 3. Run the Sweep ---
if __name__ == "__main__":
    # Initialize the sweep on wandb servers
    sweep_id = wandb.sweep(sweep_configuration, project="Deep_Bayesian_Inference_Tuning")
    
    # Run the agent (this will loop through the combinations)
    wandb.agent(sweep_id, function=train_sweep)