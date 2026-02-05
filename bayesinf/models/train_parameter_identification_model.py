import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)
from numba import cuda

if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax

    jax.default_device('cpu')
import jax.random as jr
import equinox as eqx
import optax
from functools import partial
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

from bayesinf.utils.plot_results import plot_parameter_regression_comparison
from bayesinf.utils.generate_data import normalize_data, denormalize_data, mass_spring_ode, mass_spring_ode_l, mass_spring_ode_f, mass_spring_ode2

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--flow_type", type=str, default="DDPM")
parser.add_argument('--refine', nargs='?', const='MCMC', default=None, type=str)
parser.add_argument("--n_coupling", type=int, default=8)
parser.add_argument("--n_stochastic", type=int, action="append")
parser.add_argument('--mcmc_step_size', type=float, default=5e-3)
parser.add_argument('--num_mh_steps', type=int, default=1)
parser.add_argument('--warmup_before_mcmc', type=int, default=100)
parser.add_argument("--dt0", type=float, default=0.01)
parser.add_argument("--t1", type=float, default=1.0)
parser.add_argument("--noise_std", type=float, default=0.0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--lr_scheduler", action="store_true")
parser.add_argument("--grad_norm", type=float, default=0.0)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--max_epochs", type=int, default=10000)
parser.add_argument("--force", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--multi_modal", action="store_true")
parser.add_argument("--limit", action="store_true")
parser.add_argument("--misspecification", action="store_true")
parser.add_argument("--non_linear_scale", type=float, default=1.0)
parser.add_argument("--ood", action="store_true")
parser.add_argument("--shift_scale", type=float, default=1.0)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--retrain_cn_only", action="store_true", help="Freeze the DGM and only train the condition network")
args = parser.parse_args()

train = args.train
test = args.test
flow_type = args.flow_type
dt0 = args.dt0
t1 = args.t1
noise_std = args.noise_std
force = args.force
linear = args.linear
multi_modal = args.multi_modal

STOCHASTIC_TYPES = {'MALA', 'RMALA', 'MCMC', 'BMCMC'}
FLOW_TYPES = {'REALNVP', 'NSF', 'FM', 'SBM', 'BNN', 'VAE', 'DDPM'}

if flow_type.upper() in STOCHASTIC_TYPES:
    n_stochastic = args.n_stochastic
    mcmc_step_size = args.mcmc_step_size
    num_mh_steps = args.num_mh_steps
    if isinstance(n_stochastic, int):
        name_stochastic = str(n_stochastic)
    elif isinstance(n_stochastic, list):
        name_stochastic = ""
        for i, n in enumerate(n_stochastic):
            if i < len(n_stochastic) - 1:
                name_stochastic += f"{n}_"
            else:
                name_stochastic += f"{n}"
    else:
        name_stochastic = ""
    name_stochastic += f'_{mcmc_step_size}'+f'_X{num_mh_steps}'
elif flow_type.upper() in FLOW_TYPES:
    if args.refine is not None:
        n_stochastic = 0
        name_stochastic = 'refine_'
        mcmc_step_size = args.mcmc_step_size
        num_mh_steps = args.num_mh_steps
        name_stochastic += f'{args.refine}_{mcmc_step_size}'+f'_X{num_mh_steps}'
    else:
        name_stochastic = ""
else:
    name_stochastic = ""


data_set = f"mass_spring_damper_{dt0}_{t1}"
if noise_std > 0:
    data_set += f'_noise_{noise_std}'
if force:
    data_set += '_force'
if linear:
    data_set += '_linear'
if multi_modal:
    data_set += '_multi_modal'

if not test:
    import wandb
    if name_stochastic != "": 
        wandb.init(project='Deep_Bayesian_Inference', name=f'{flow_type}_{name_stochastic}_{data_set}')
    else:
        if args.retrain:
            suffix = "retrain_cn" if args.retrain_cn_only else "retrain"
            if args.misspecification:
                wandb.init(project='Deep_Bayesian_Inference', name= f"{flow_type}_{data_set}_misspecification_{args.non_linear_scale}_{suffix}")
            elif args.ood:
                wandb.init(project='Deep_Bayesian_Inference', name= f"{flow_type}_{data_set}_ood_{args.shift_scale}_{suffix}")
            else:
                wandb.init(project='Deep_Bayesian_Inference', name= f"{flow_type}_{data_set}_{suffix}")
        else:
            wandb.init(project='Deep_Bayesian_Inference', name= f"{flow_type}_{data_set}")

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
            if args.misspecification:
                _generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std', 'beta', 'alpha'])(
                                         generate_mass_spring_data_f)
            elif args.ood:
                _generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std', 'shift'])(
                                         generate_mass_spring_data_f)                            
            else:
                _generate_data_jit = None
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

def compute_ensemble_likelihood_fast(theta, y_obs, n_ahead, ode_type, sigma):
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

def log_posterior_fn(theta, y_obs, gradient_scale=0.0):
    n_ahead = 5
    sigma = max(0.01, noise_std)
    
    # Determine ODE type string
    ode_type = 'multi_modal' if multi_modal else ('linear' if linear else ('force' if force else 'standard'))
    comp_traj = lambda t, y : compute_ensemble_likelihood_fast(t, y, n_ahead, ode_type, sigma)
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  
results_dir = os.path.join(project_root, 'results')
if name_stochastic != "":
    save_path_pt = os.path.join(results_dir, f"{flow_type}_{name_stochastic}_{data_set}.pt")
    save_path_png = os.path.join(results_dir, f"{flow_type}_{name_stochastic}_{data_set}.png")
else:
    save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}.pt")
    if args.retrain:
        suffix = "retrain_cn" if args.retrain_cn_only else "retrain"
        if args.misspecification:
            save_path_png = os.path.join(results_dir, f"{flow_type}_{data_set}_misspecification_{args.non_linear_scale}_{suffix}.png")
        elif args.ood:
            save_path_png = os.path.join(results_dir, f"{flow_type}_{data_set}_ood_{args.shift_scale}_{suffix}.png")
        else:
            save_path_png = os.path.join(results_dir, f"{flow_type}_{data_set}_{suffix}.png")
    else:
        save_path_png = os.path.join(results_dir, f"{flow_type}_{data_set}.png")

feature_dim = 2 
hidden_dim_rnn = 64  # GRU hidden size
feature_latent_dim = 64
hidden_dim_time = 4
hidden_sizes = [64] * 2
param_dim = 2 if multi_modal else 3  # Number of physical parameters to infer (theta_p)
param_names = [r'$k_1$', r'$k_3$'] if multi_modal else [r'$m$', r'$k$', r'$c$']
batch_size = 256  # batch_size of synthetic training data points

train_key = jr.PRNGKey(5)
if args.limit:
    train_size = batch_size
    if args.misspecification:
        train_D, train_theta_p = _generate_data_jit(train_key, dt0=dt0, t1=t1, n_samples=train_size, noise_std=noise_std, alpha=1.0*args.non_linear_scale, beta=1.0*args.non_linear_scale)
    elif args.ood:
        train_D, train_theta_p = _generate_data_jit(train_key, dt0=dt0, t1=t1, n_samples=train_size, noise_std=noise_std, shift=(0.1*args.shift_scale, 1*args.shift_scale, 0.5*args.shift_scale))
    else:
        train_D, train_theta_p = generate_data_jit(train_key, dt0=dt0, t1=t1, n_samples=train_size, noise_std=noise_std)
    @jax.jit
    def shuffle_batch(x, y, key):
        perm = jax.random.permutation(key, x.shape[0])
        return x[perm], y[perm]

test_key = jr.PRNGKey(0)
test_key, sample_key = jr.split(test_key, 2)
test_size = 200
n_samples = 100
test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
test_D, mean, std = normalize_data(test_D)
test_theta_p_norm, mean_theta, std_theta = normalize_data(test_theta_p)

if args.misspecification:
    test_D, test_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, beta=1*args.non_linear_scale, alpha=1*args.non_linear_scale)
    test_D = normalize_data(test_D, mean, std)
elif args.ood:
    test_D, test_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, shift=(0.1*args.shift_scale, 1*args.shift_scale, 0.5*args.shift_scale))
    test_D = normalize_data(test_D, mean, std)
else:
    pass

patience = args.patience
if patience > 0:
    val_key = jr.PRNGKey(2)
    val_size = batch_size
    if args.misspecification:
        val_D, val_theta_p = _generate_data_jit(val_key, dt0=dt0, t1=t1, n_samples=val_size, noise_std=noise_std, alpha=1.0*args.non_linear_scale, beta=1.0*args.non_linear_scale)
    elif args.ood:
        val_D, val_theta_p = _generate_data_jit(val_key, dt0=dt0, t1=t1, n_samples=val_size, noise_std=noise_std, shift=(0.1*args.shift_scale, 1*args.shift_scale, 0.5*args.shift_scale))
    else:
        val_D, val_theta_p = generate_data_jit(val_key, dt0=dt0, t1=t1, n_samples=val_size, noise_std=noise_std)
    val_D = normalize_data(val_D, mean, std)
    val_theta_p = normalize_data(val_theta_p, mean_theta, std_theta)

model_key = jr.PRNGKey(42)

# Initialize Model
if flow_type.upper() == 'BNN':
    from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator, make_kl_annealing_schedule, make_step, batched_elbo_loss
    kl_weight = 1e-3
    kl_weight_scheduler = make_kl_annealing_schedule(0, 50, kl_weight)
    loss_fn = lambda m, y, x, k: batched_elbo_loss(m, y, x, k, kl_weight=kl_weight)
    model = BNNPosteriorEstimator(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                  key=model_key, noise_std=noise_std, mcmc_type=args.refine if args.refine else 'MCMC' if args.refine else 'MCMC',
                                  mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                                  log_posterior_fn=log_posterior_fn if not args.refine else None)
elif flow_type.upper() == 'VAE':
    from bayesinf.models.variational_autoencoder import cVAE, make_step, cvae_loss
    loss_fn = lambda m, y, x, k: cvae_loss(m, y, x, k)
    model = cVAE(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                 key=model_key, mcmc_type=args.refine if args.refine else 'MCMC', noise_std=noise_std, 
                 mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                 log_posterior_fn=log_posterior_fn if not args.refine else None)
elif flow_type.upper() == 'DDPM':
    from bayesinf.models.denoising_diffusion_model import ConditionalDDPM, make_step
    loss_fn = lambda m, y, x, k: m.loss(y, x, k)
    model = ConditionalDDPM(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                            hidden_sizes, key=model_key, mcmc_type=args.refine if args.refine else 'MCMC', noise_std=noise_std, 
                            mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                            log_posterior_fn=log_posterior_fn if not args.refine else None)
elif flow_type.upper() == 'SBM':
    from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel, make_step
    loss_fn = lambda m, y, x, k: m.loss(y, x, k)
    model = ConditionalScoreModel(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                  hidden_sizes, key=model_key, mcmc_type=args.refine if args.refine else 'MCMC', noise_std=noise_std, 
                                  mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                                  log_posterior_fn=log_posterior_fn if not args.refine else None)
elif flow_type.upper() == 'FM':
    from bayesinf.models.flow_matching import ConditionalFlowMatching, make_step, cfm_loss
    loss_fn = lambda m, y, x, k: cfm_loss(m, y, x, k)
    model = ConditionalFlowMatching(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                    key=model_key, mcmc_type=args.refine if args.refine else 'MCMC', noise_std=noise_std, 
                                    mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                                    log_posterior_fn=log_posterior_fn if not args.refine else None)
elif flow_type.upper() in STOCHASTIC_TYPES:
    from bayesinf.models.MCMC_Flow import CondSNF, make_step, snf_loss
    loss_fn = lambda m, y, x, k: snf_loss(m, y, x, k)
    model = CondSNF(param_dim, feature_dim, log_posterior_fn, key=model_key, n_coupling=args.n_coupling, 
                    n_stochastic=n_stochastic, hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                    encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, rnn_type='lstm', 
                    flow_type='realnvp', mcmc_type=flow_type, noise_std=noise_std, 
                    mcmc_step_size=args.mcmc_step_size, num_mh_steps=args.num_mh_steps, 
                    apply_gradient_schedule=False)  
    def get_gradient_scale(epoch, warmup=100, transition=50):
        if epoch < warmup:
            return 0.0
        elif epoch >= warmup + transition:
            return 1.0
        else:
            progress = (epoch - warmup) / transition
            return 0.5 * (1 - jnp.cos(jnp.pi * progress))  # Cosine schedule
else:
    from bayesinf.models.normalizing_flow import CondNormFlow, make_step, cnf_loss
    loss_fn = lambda m, y, x, k: cnf_loss(m, y, x)
    model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=args.n_coupling, 
                         hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                         encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                         rnn_type='lstm', flow_type=flow_type)                                    

if train:
    if args.retrain:
        model = eqx.tree_deserialise_leaves(save_path_pt, model)
        suffix = "retrain_cn" if args.retrain_cn_only else "retrain"
        if args.misspecification:
            save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}_misspecification_{args.non_linear_scale}_{suffix}.pt")
        elif args.ood:
            save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}_ood_{args.shift_scale}_{suffix}.pt")
        else:
            save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}_{suffix}.pt")
    # Optimization Setup
    num_epochs = args.max_epochs
    learning_rate = args.learning_rate
    if args.grad_norm > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(args.grad_norm), 
                                optax.adamw(learning_rate=args.learning_rate, 
                                weight_decay=1e-4))
    else:
        if args.lr_scheduler:
            schedule_fn = optax.join_schedules(
                            schedules=[
                                optax.constant_schedule(learning_rate),
                                optax.exponential_decay(learning_rate, 
                                      transition_steps=100, decay_rate=0.9)
                            ],
                            boundaries=[200] # Switch at step 1000
                        )
            optimizer = optax.adam(schedule_fn)
        else:
            optimizer = optax.adam(learning_rate)

    if args.retrain_cn_only:

        filter_spec = jax.tree_util.tree_map(lambda _: False, model)
        
        if hasattr(model, 'encoder'):
            filter_spec = eqx.tree_at(lambda m: m.encoder, filter_spec, 
                                    jax.tree_util.tree_map(lambda _: True, model.encoder))
        elif hasattr(model, 'ts_encoder'):
            filter_spec = eqx.tree_at(lambda m: m.ts_encoder, filter_spec, 
                                    jax.tree_util.tree_map(lambda _: True, model.ts_encoder))

        trainable_filter = jax.tree_util.tree_map(lambda a, b: a and eqx.is_inexact_array(b), filter_spec, model)
    else:
        trainable_filter = eqx.is_inexact_array

    opt_state = optimizer.init(eqx.filter(model, trainable_filter))

    if patience > 0:
        best_model = model
        best_val_loss = np.inf
        patience_counter = 0

    # Training Loop
    print(f"Starting {flow_type} Training...Max. Epochs: {num_epochs}")

    with tqdm(range(num_epochs), desc='step: ') as pstep:
        for i in pstep:
            data_key, train_key, val_key = jr.split(train_key, 3)
            if not args.limit:
                if args.retrain:
                    if args.misspecification:
                        batch_D, batch_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, beta=1, alpha=1)
                    elif args.ood:
                            batch_D, batch_theta_p = _generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std, shift=(0.1, 1, 0.5))
                    else:
                        batch_D, batch_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)
                else:
                    batch_D, batch_theta_p = generate_data_jit(data_key, dt0=dt0, t1=t1, n_samples=batch_size, noise_std=noise_std)
            else:
                batch_D, batch_theta_p = shuffle_batch(train_D, train_theta_p, data_key)
            batch_D = normalize_data(batch_D, mean, std)
            batch_theta_p = normalize_data(batch_theta_p, mean_theta, std_theta)
            if flow_type.upper() in STOCHASTIC_TYPES:
                gradient_scale = get_gradient_scale(i, warmup=args.warmup_before_mcmc, transition=50)
                model, opt_state, loss = make_step(model, batch_D, batch_theta_p, train_key, opt_state, optimizer, trainable_filter if args.retrain_cn_only else None,
                                                   gradient_scale=gradient_scale)
            else:
                model, opt_state, loss = make_step(model, batch_D, batch_theta_p, train_key, opt_state, optimizer, trainable_filter if args.retrain_cn_only else None)
            if patience > 0:
                val_loss = loss_fn(model, val_D, val_theta_p, val_key)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    eqx.tree_serialise_leaves(save_path_pt, best_model)
                    inference_model = eqx.nn.inference_mode(model)
                    if name_stochastic != "":
                        if n_stochastic == 0 or 0 in n_stochastic or args.refine:
                            test_theta_samples = inference_model.sample_with_MCMC(test_D, sample_key, n_samples)
                        else:
                            test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
                    else:
                        test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
                    test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])

                    plot_parameter_regression_comparison(test_theta_p, test_theta_samples, param_names, 
                                                        n_rows=1 if force else 3, n_cols=2 if multi_modal else 3,
                                                        filename=save_path_png if test else None, 
                                                        correlated= None if force or multi_modal else [[0, 1], [0, 2]],
                                                        indices=[0, 49, 99, 199] if multi_modal else (None if force else [0, 199]),
                                                        wandb_log=not test, lite=True)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {i}")
                    break
            else:
                if i % 100 == 0:
                    eqx.tree_serialise_leaves(save_path_pt, model)
                    inference_model = eqx.nn.inference_mode(model)
                    if name_stochastic != "":
                        if n_stochastic == 0 or 0 in n_stochastic or args.refine:
                            test_theta_samples = inference_model.sample_with_MCMC(test_D, sample_key, n_samples)
                        else:
                            test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
                    else:
                        test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
                    test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])

                    plot_parameter_regression_comparison(test_theta_p, test_theta_samples, param_names, 
                                                        n_rows=1 if force else 3, n_cols=2 if multi_modal else 3,
                                                        filename=save_path_png if test else None, 
                                                        correlated= None if force or multi_modal else [[0, 1], [0, 2]],
                                                        indices=[0, 49, 99, 199] if multi_modal else (None if force else [0, 199]),
                                                        wandb_log=not test, lite=True)

            if not test:
                wandb.log({'loss': loss})
                if patience > 0:
                    wandb.log({'val_loss': val_loss})

else:
    if args.retrain:
        save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}_retrain.pt")
    model = eqx.tree_deserialise_leaves(save_path_pt, model)
    inference_model = eqx.nn.inference_mode(model)
    if name_stochastic != "":
        if n_stochastic == 0 or 0 in n_stochastic or args.refine:
            test_theta_samples = inference_model.sample_with_MCMC(test_D, sample_key, n_samples)
        else:
            test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
    else:
        test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
    test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])

    plot_parameter_regression_comparison(test_theta_p, test_theta_samples, param_names, 
                                         n_rows=1 if force else 3, n_cols=2 if multi_modal else 3,
                                         filename=save_path_png, 
                                         correlated= None if force or multi_modal else [[0, 1], [0, 2]],
                                         indices=[0, 49, 99, 199] if multi_modal else (None if force else [0, 199]),
                                         wandb_log=False)