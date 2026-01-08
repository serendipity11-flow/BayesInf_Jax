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
import diffrax as dfx
import optax
from functools import partial
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

from bayesinf.utils.plot_results import plot_parameter_regression_comparison
from bayesinf.utils.generate_data import normalize_data, denormalize_data

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--flow_type", default="DDPM")
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

data_set = f"mass_spring_damper_{dt0}_{t1}"
if noise_std > 0:
    data_set += f'_noise_{noise_std}'
if force:
    data_set += '_force'
if linear:
    data_set += '_linear'
if multi_modal:
    data_set += '_multi_modal'
if args.limit:
    data_set += '_limit'

if not test:
    import wandb
    wandb.init(project='Deep_Bayesian_Inference', name=f'{flow_type}_{data_set}')

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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  
results_dir = os.path.join(project_root, 'results')
save_path_pt = os.path.join(results_dir, f"{flow_type}_{data_set}.pt")
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

if multi_modal:
    # dynamics_fn is actually the solution function
    def dynamics_fn(y0, theta, ts):
        _mean0 = jnp.stack([mean[..., 0]//2, mean[..., 1]//2, mean[..., 0]//2, mean[..., 1]//2], axis=-1)
        _std0 = jnp.stack([std[..., 0]//2, std[..., 1]//2, std[..., 0]//2, std[..., 1]//2], axis=-1)
        y0 = denormalize_data(y0, jnp.squeeze(_mean0), jnp.squeeze(_std0))
        theta = denormalize_data(theta, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
        sol = dfx.diffeqsolve(dfx.ODETerm(mass_spring_ode2), dfx.Dopri5(), t0=0.0, t1=t1-dt0, dt0=dt0, 
                              y0=y0, args=(theta,), saveat=dfx.SaveAt(ts=ts)).ys
        sol = jax.lax.stop_gradient(sol)
        state = jnp.stack([sol[:, 0] + sol[:, 2], sol[:, 1] + sol[:, 3]], axis=-1)
        state = normalize_data(state, mean[0], std[0])
        return state

else:
    if linear:
        def dynamics_fn(t, state_norm, theta_norm):
            state = denormalize_data(state_norm, jnp.squeeze(mean), jnp.squeeze(std))
            theta = denormalize_data(theta_norm, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
            deriv = jax.vmap(lambda _t, _s: mass_spring_ode_l(_t, _s, (theta,)))(t, state)
            deriv_norm = normalize_data(deriv, jnp.squeeze(mean), jnp.squeeze(std))
            return deriv_norm
    else:
        if force:
            def dynamics_fn(t, state_norm, theta_norm):
                state = denormalize_data(state_norm, jnp.squeeze(mean), jnp.squeeze(std))
                theta = denormalize_data(theta_norm, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
                deriv = jax.vmap(lambda _t, _s: mass_spring_ode_f(_t, _s, (theta,)))(t, state)
                deriv_norm = normalize_data(deriv, jnp.squeeze(mean), jnp.squeeze(std))
                return deriv_norm
        else:
            def dynamics_fn(t, state_norm, theta_norm):
                state = denormalize_data(state_norm, jnp.squeeze(mean), jnp.squeeze(std))
                theta = denormalize_data(theta_norm, jnp.squeeze(mean_theta), jnp.squeeze(std_theta))
                deriv = jax.vmap(lambda _t, _s: mass_spring_ode(_t, _s, (theta,)))(t, state)
                deriv_norm = normalize_data(deriv, jnp.squeeze(mean), jnp.squeeze(std))
                return deriv_norm

model_key = jr.PRNGKey(42)

# Initialize Model
if flow_type.upper() == 'BNN':
    from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator, make_kl_annealing_schedule, make_step
    kl_weight = 1e-3
    kl_weight_scheduler = make_kl_annealing_schedule(0, 50, kl_weight)
    model = BNNPosteriorEstimator(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                  key=model_key)
elif flow_type.upper() == 'VAE':
    from bayesinf.models.variational_autoencoder import cVAE, make_step
    model = cVAE(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                 key=model_key)
elif flow_type.upper() == 'DDPM':
    from bayesinf.models.denoising_diffusion_model import ConditionalDDPM, make_step
    model = ConditionalDDPM(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                            hidden_sizes, key=model_key)
elif flow_type.upper() == 'SBM':
    from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel, make_step
    model = ConditionalScoreModel(feature_dim, param_dim, feature_latent_dim, hidden_dim_rnn, hidden_dim_time,
                                  hidden_sizes, key=model_key)
elif flow_type.upper() == 'FM':
    from bayesinf.models.flow_matching import ConditionalFlowMatching, make_step
    model = ConditionalFlowMatching(feature_dim, param_dim, feature_latent_dim, hidden_sizes, hidden_dim_rnn,
                                    key=model_key)
elif flow_type.upper() == 'MALA':
    from bayesinf.models.MCMC_Flow import CondSNF, make_step
    def log_posterior_fn(x, y):
        num_steps = int(t1/dt0)
        ts = jnp.linspace(0, t1-dt0, num_steps)
        if x.ndim > 1:
            states = y[:, 1:] - y[:, :-1]
        else:
            states = y[1:] - y[:-1]
        if multi_modal:
            if x.ndim > 1:
                y0 = jnp.stack([y[:, 0, 0]/2, jnp.zeros((y.shape[0],)), y[:, 0, 0]/2, jnp.zeros((y.shape[0],))], axis=-1)
                pred = jax.vmap(dynamics_fn, in_axes=(0, 0, None))(y0, x, ts)
            else:
                y0 = jnp.array([y[0, 0]/2, 0, y[0, 0]/2, 0])
                pred = dynamics_fn(y0, x, ts)
            residual = pred - states
        else:
            states_deriv = states / dt0
            if x.ndim > 1:
                pred_deriv = jax.vmap(dynamics_fn, in_axes=(None, 0, 0))(ts, states, x)
            else:
                pred_deriv = dynamics_fn(ts, states, x)
            residual = pred_deriv - states_deriv
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
    from bayesinf.models.normalizing_flow import CondNormFlow, make_step
    model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, 
                         hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
                         encoded_latent_dim=feature_latent_dim, hidden_rnn=hidden_dim_rnn, 
                         rnn_type='lstm', flow_type=flow_type)                                    

if train:
    # Optimization Setup
    num_epochs = args.max_epochs
    patience = args.patience
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
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model = model
    best_val_loss = np.inf
    patience_counter = 0

    # Training Loop
    print(f"Starting {flow_type} Training...Max. Epochs: {num_epochs}")

    with tqdm(range(num_epochs), desc='step: ') as pstep:
        for i in pstep:
            data_key, train_key = jr.split(train_key)
            if not args.limit:
                batch_D, batch_theta_p = generate_data_jit(data_key, dt0=dt0, t1=t1, n_samples=batch_size, noise_std=noise_std)
            else:
                batch_D, batch_theta_p = shuffle_batch(train_D, train_theta_p, data_key)
            batch_D = normalize_data(batch_D, mean, std)
            batch_theta_p = normalize_data(batch_theta_p, mean_theta, std_theta)
            model, opt_state, loss = make_step(model, batch_D, batch_theta_p, train_key, opt_state, optimizer)
            if loss < best_val_loss:
                best_val_loss = loss
                best_model = model
                eqx.tree_serialise_leaves(save_path_pt, best_model)
                inference_model = eqx.nn.inference_mode(model)
                test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
                test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])

                plot_parameter_regression_comparison(test_theta_p, test_theta_samples, param_names, 
                                                     n_rows=1 if force else 3, n_cols=2 if multi_modal else 3,
                                                     filename=save_path_png if test else None, 
                                                     correlated= None if force or multi_modal else [[0, 1], [0, 2]],
                                                     indices=[0, 49, 99, 199] if multi_modal else (None if force else [0, 199]),
                                                     wandb_log=not test)
                patience_counter = 0
            else:
                patience_counter += 1

            if not test:
                wandb.log({'loss': loss})

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {i}")
                break
else:
    model = eqx.tree_deserialise_leaves(save_path_pt, model)
    inference_model = eqx.nn.inference_mode(model)
    test_theta_samples = inference_model.sample(test_D, sample_key, n_samples)
    test_theta_samples = denormalize_data(test_theta_samples, mean_theta[None], std_theta[None])

    plot_parameter_regression_comparison(test_theta_p, test_theta_samples, param_names, 
                                         n_rows=1 if force else 3, n_cols=2 if multi_modal else 3,
                                         filename=save_path_png, 
                                         correlated= None if force or multi_modal else [[0, 1], [0, 2]],
                                         indices=[0, 49, 99, 199] if multi_modal else (None if force else [0, 199]),
                                         wandb_log=False)