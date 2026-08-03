"""Train a controlled RealNVP FlowSDE on ALOHA Mobile Cabinet transitions."""

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .controlled_flow_sde import ControlledRealNVPFlowSDE
from .second_order_flow_sde import SecondOrderControlledRealNVPFlowSDE
from .aloha_data import (
    augment_with_causal_velocity, load_episodes, load_transition_auxiliary,
    normalization,
    split_episode_indices,
)
from .pusher_geometry import pusher_contact_features

ARM_INDICES = jnp.asarray([0, 1, 2, 6, 7, 8])
OBJECT_INDICES = jnp.asarray([3, 4, 5, 9, 10, 11])


def contact_temporal_weights(forces, scale):
    """Smooth positive trajectory weights derived from training-only wrench labels."""
    raw = jnp.sqrt(jnp.mean((forces / jnp.asarray(scale)) ** 2, axis=-1))
    standardized = (raw - jnp.mean(raw)) / (jnp.std(raw) + 1e-6)
    weights = jax.nn.softplus(standardized)
    return weights / (jnp.mean(weights) + 1e-6)


def transition_arrays(episodes, actions, indices, x_mean, x_std, u_mean, u_std,
                      dt=0.02):
    """Flatten complete episodes without creating cross-episode transitions."""
    x0, x1, u0, times = [], [], [], []
    for index in indices:
        x = (episodes[index] - np.asarray(x_mean)) / np.asarray(x_std)
        u = (actions[index] - np.asarray(u_mean)) / np.asarray(u_std)
        x0.append(x[:-1]); x1.append(x[1:]); u0.append(u)
        times.append(np.arange(len(u), dtype=np.float32) * dt)
    return tuple(jnp.asarray(np.concatenate(values)) for values in (x0, u0, x1, times))


def trajectory_windows(episodes, actions, indices, x_mean, x_std, u_mean, u_std,
                       window_steps=128, stride=64, auxiliary=None):
    """Create episode-contained windows for stochastic multiple shooting."""
    states, controls, starts, labels = [], [], [], []
    for index in indices:
        x = (episodes[index] - np.asarray(x_mean)) / np.asarray(x_std)
        u = (actions[index] - np.asarray(u_mean)) / np.asarray(u_std)
        for start in range(0, len(u) - window_steps + 1, stride):
            states.append(x[start:start + window_steps + 1])
            controls.append(u[start:start + window_steps])
            starts.append(start)
            if auxiliary is not None:
                if len(auxiliary[index]) != len(actions[index]):
                    raise ValueError("auxiliary labels must align with transitions")
                labels.append(auxiliary[index][start:start + window_steps])
    if not states:
        raise ValueError("window configuration produced no training trajectories")
    result = (jnp.asarray(np.stack(states)), jnp.asarray(np.stack(controls)),
              jnp.asarray(starts, dtype=jnp.int32))
    if auxiliary is not None:
        result += (jnp.asarray(np.stack(labels)),)
    return result


def build_model(args, state_dim, control_dim):
    if (getattr(args, "velocity_augmented", False)
            or getattr(args, "state_is_position_velocity", False)):
        if state_dim % 2:
            raise ValueError("velocity-augmented state dimension must be even")
        return SecondOrderControlledRealNVPFlowSDE(
            state_dim // 2, control_dim, jr.PRNGKey(args.seed + 1),
            hidden_size=args.width, depth=args.depth,
            diffusion_hidden=args.diffusion_width,
            flow_layers=args.flow_layers, time_emb_dim=args.time_embedding,
            realnvp_alpha=args.realnvp_alpha, dt=args.dt,
            velocity_to_position_scale=args.velocity_to_position_scale,
            velocity_to_position_bias=args.velocity_to_position_bias,
            diffusion_condition_on_control=not getattr(
                args, "diffusion_no_control", False
            ),
        )
    return ControlledRealNVPFlowSDE(
        state_dim, control_dim, jr.PRNGKey(args.seed + 1),
        hidden_size=args.width, depth=args.depth,
        diffusion_hidden=args.diffusion_width, flow_layers=args.flow_layers,
        time_emb_dim=args.time_embedding, realnvp_alpha=args.realnvp_alpha,
        dt=args.dt,
        diffusion_condition_on_control=not getattr(
            args, "diffusion_no_control", False
        ),
        object_drift_no_control=getattr(args, "object_drift_no_control", False),
        separate_dynamics_heads=getattr(args, "separate_dynamics_heads", False),
        use_contact_geometry=getattr(args, "use_contact_geometry", False),
        geometry_temperature=getattr(args, "geometry_temperature", 0.02),
        state_mean=getattr(args, "model_state_mean", None),
        state_std=getattr(args, "model_state_std", None),
        geometry_mean=getattr(args, "geometry_mean", None),
        geometry_std=getattr(args, "geometry_std", None),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if not hasattr(args, "object_trajectory_supervision_weight"):
        args.object_trajectory_supervision_weight = 0.0
    if args.contact_weight <= 0:
        raise ValueError("--contact-weight must be positive")
    if not 0.0 <= args.contact_center <= 1.0:
        raise ValueError("--contact-center must be between 0 and 1")
    if args.contact_width <= 0:
        raise ValueError("--contact-width must be positive")
    if args.contact_force_supervision_weight < 0:
        raise ValueError("--contact-force-supervision-weight must be non-negative")
    if args.object_trajectory_supervision_weight < 0:
        raise ValueError("--object-trajectory-supervision-weight must be non-negative")
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    contact_forces = None
    if (args.contact_force_supervision_weight > 0
            or args.object_trajectory_supervision_weight > 0):
        if state_dim < 12:
            raise ValueError(
                "contact-force supervision requires [q1:q6, v1:v6] state"
            )
        contact_forces = load_transition_auxiliary(
            args.data, args.contact_force_column, episode_ids
        )
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            anonymous="allow",
            config={
                "seed": args.seed, "epochs": args.epochs,
                "batch_size": args.batch_size, "learning_rate": args.lr,
                "dt": args.dt, "width": args.width, "depth": args.depth,
                "lr_decay_epoch": args.lr_decay_epoch,
                "lr_after_decay": args.lr_after_decay,
                "diffusion_width": args.diffusion_width,
                "flow_layers": args.flow_layers,
                "window_steps": args.window_steps,
                "segment_length": args.segment_length,
                "stride": args.stride,
                "train_particles": args.train_particles,
                "solver": args.solver, "path_weight": args.path_weight,
                "uncertainty_weight": args.uncertainty_weight,
                "continuity_weight": args.continuity_weight,
                "contact_weight": args.contact_weight,
                "contact_center": args.contact_center,
                "contact_width": args.contact_width,
                "state_weight_epoch": args.state_weight_epoch,
                "object_state_weight": args.object_state_weight,
                "contact_force_supervision_weight": (
                    args.contact_force_supervision_weight
                ),
                "object_trajectory_supervision_weight": (
                    args.object_trajectory_supervision_weight
                ),
            },
        )
    position_dim = state_dim
    if args.velocity_augmented:
        episodes = augment_with_causal_velocity(
            episodes, dt=args.dt, window=args.velocity_window
        )
        state_dim = 2 * position_dim
    elif args.state_is_position_velocity:
        if state_dim % 2:
            raise ValueError("existing [q,v] state dimension must be even")
        position_dim = state_dim // 2
    if args.state_loss_weights:
        state_loss_weights = jnp.asarray(
            [float(value) for value in args.state_loss_weights.split(",")]
        )
        if state_loss_weights.shape != (state_dim,):
            raise ValueError(
                f"--state-loss-weights needs {state_dim} values, got "
                f"{state_loss_weights.shape[0]}"
            )
    else:
        state_loss_weights = jnp.ones((state_dim,))
    train_i, val_i, test_i = split_episode_indices(len(episodes), args.seed)
    if len(test_i) == 0:
        raise ValueError("dataset needs enough episodes for a held-out test split")
    stats = normalization(episodes, actions, train_i)
    if args.use_contact_geometry:
        train_states = np.concatenate([episodes[i] for i in train_i])
        geometry = np.asarray(jax.vmap(
            lambda state: pusher_contact_features(
                state, temperature=args.geometry_temperature
            )
        )(jnp.asarray(train_states)))
        args.geometry_mean = geometry.mean(0).tolist()
        args.geometry_std = np.maximum(geometry.std(0), 1e-4).tolist()
    else:
        args.geometry_mean, args.geometry_std = [0.0] * 5, [1.0] * 5
    args.model_state_mean = np.asarray(stats[0]).tolist()
    args.model_state_std = np.asarray(stats[1]).tolist()
    contact_force_scale = None
    if contact_forces is not None:
        train_contact = np.concatenate([contact_forces[i] for i in train_i])
        contact_force_scale = np.maximum(
            np.sqrt(np.mean(train_contact ** 2, axis=0)), 1e-6
        ).astype(np.float32)
    if args.velocity_augmented or args.state_is_position_velocity:
        # In normalized coordinates, dq_n/dt = scale*v_n + bias.
        q_std = np.asarray(stats[1][:position_dim])
        v_mean = np.asarray(stats[0][position_dim:])
        v_std = np.asarray(stats[1][position_dim:])
        args.velocity_to_position_scale = (v_std / q_std).tolist()
        args.velocity_to_position_bias = (v_mean / q_std).tolist()
    if args.window_steps % args.segment_length:
        raise ValueError("--window-steps must be divisible by --segment-length")
    train = trajectory_windows(
        episodes, actions, train_i, *stats, args.window_steps, args.stride,
        auxiliary=contact_forces,
    )
    validation = trajectory_windows(
        episodes, actions, val_i, *stats, args.window_steps, args.window_steps,
        auxiliary=contact_forces,
    )
    test = transition_arrays(episodes, actions, test_i, *stats, dt=args.dt)
    model = build_model(args, state_dim, control_dim)
    effective_windows = (
        min(args.samples_per_epoch, len(train[0]))
        if args.samples_per_epoch > 0 else len(train[0])
    )
    steps_per_epoch = int(np.ceil(effective_windows / args.batch_size))
    learning_rate = optax.piecewise_constant_schedule(
        init_value=args.lr,
        boundaries_and_scales={
            args.lr_decay_epoch * steps_per_epoch: args.lr_after_decay / args.lr
        },
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=1e-5),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    base_state_loss_weights = state_loss_weights
    if contact_forces is not None:
        # Privileged objectives exclusively supervise object path error here;
        # generative transition NLL still identifies the full diffusion.
        base_state_loss_weights = base_state_loss_weights.at[OBJECT_INDICES].set(0.0)

    @eqx.filter_jit
    def train_step(current_model, current_state, batch, noise_key,
                   current_state_loss_weights, contact_coefficient,
                   trajectory_coefficient):
        bx, bu, bstart = batch[:3]
        bcontact = batch[3] if len(batch) == 4 else None
        def components(m):
            path_and_continuity = m.multiple_shooting_loss(
                bx, bu, bstart, noise_key,
                segment_length=args.segment_length,
                n_particles=args.train_particles, solver_name=args.solver,
                shooting_noise_std=args.state_noise_std,
                control_noise_std=args.control_noise_std,
                path_weight=args.path_weight,
                uncertainty_weight=args.uncertainty_weight,
                continuity_weight=args.continuity_weight,
                state_loss_weights=current_state_loss_weights,
                contact_weight=args.contact_weight,
                contact_center=args.contact_center,
                contact_width=args.contact_width,
            )
            times = (
                bstart[:, None] + jnp.arange(bu.shape[1])[None, :]
            ) * args.dt
            transition = m.noisy_transition_nll(
                bx[:, :-1], bu, bx[:, 1:], times, jr.fold_in(noise_key, 71),
                state_noise_std=args.state_noise_std,
                control_noise_std=args.control_noise_std,
            )
            times_flat = times.reshape(-1)
            x_flat = bx[:, :-1].reshape(-1, state_dim)
            u_flat = bu.reshape(-1, control_dim)
            target_flat = bx[:, 1:].reshape(-1, state_dim)
            weighted_mse = jnp.asarray(0.0)
            if args.state_mse_weight != 0:
                drift = jax.vmap(m.drift)(x_flat, u_flat, times_flat)
                predicted = x_flat + args.dt * drift
                weighted_mse = jnp.mean(
                    current_state_loss_weights * (predicted - target_flat) ** 2
                )
            contact_supervision = jnp.asarray(0.0)
            trajectory_supervision = jnp.asarray(0.0)
            if bcontact is not None:
                if args.state_mse_weight == 0:
                    drift = jax.vmap(m.drift)(x_flat, u_flat, times_flat)
                    predicted = x_flat + args.dt * drift
                force = bcontact.reshape(-1, bcontact.shape[-1])
                intensity = contact_temporal_weights(
                    force, contact_force_scale
                )
                object_error = jnp.mean(
                    (predicted[:, jnp.asarray([3, 4, 5, 9, 10, 11])]
                     - target_flat[:, jnp.asarray([3, 4, 5, 9, 10, 11])]) ** 2,
                    axis=-1,
                )
                contact_supervision = jnp.sum(intensity * object_error) / jnp.maximum(
                    jnp.sum(intensity), 1.0
                )
                if args.object_trajectory_supervision_weight > 0:
                    object_weights = jnp.zeros((state_dim,), dtype=bx.dtype)
                    object_weights = object_weights.at[OBJECT_INDICES].set(1.0)
                    trajectory_supervision = m.multiple_shooting_loss(
                        bx, bu, bstart, jr.fold_in(noise_key, 173),
                        segment_length=args.segment_length,
                        n_particles=args.train_particles,
                        solver_name=args.solver,
                        shooting_noise_std=args.state_noise_std,
                        control_noise_std=args.control_noise_std,
                        path_weight=1.0,
                        uncertainty_weight=args.uncertainty_weight,
                        continuity_weight=args.continuity_weight,
                        state_loss_weights=object_weights,
                        temporal_loss_weights=intensity.reshape(bcontact.shape[:2]),
                        contact_weight=1.0,
                    )
            base = (path_and_continuity + args.transition_weight * transition
                    + args.state_mse_weight * weighted_mse)
            auxiliary = (contact_coefficient * contact_supervision
                         + trajectory_coefficient * trajectory_supervision)
            return base, auxiliary

        base_loss, base_grads = eqx.filter_value_and_grad(
            lambda m: components(m)[0]
        )(current_model)
        # Obtain the contact contribution independently for diagnostics and
        # combine the two gradient trees explicitly.
        contact_loss, contact_grads = eqx.filter_value_and_grad(
            lambda m: components(m)[1]
        )(current_model)
        grads = jax.tree_util.tree_map(
            lambda base, contact: (None if base is None else base + contact),
            base_grads, contact_grads, is_leaf=lambda value: value is None,
        )
        loss = base_loss + contact_loss
        base_norm = optax.global_norm(base_grads)
        contact_norm = optax.global_norm(contact_grads)
        updates, next_state = optimizer.update(
            grads, current_state, eqx.filter(current_model, eqx.is_inexact_array)
        )
        return (eqx.apply_updates(current_model, updates), next_state, loss,
                base_norm, contact_norm)

    best_model, best_validation = model, float("inf")
    curve, started = [], time.perf_counter()
    for epoch in range(args.epochs):
        current_state_loss_weights = base_state_loss_weights
        if epoch >= args.state_weight_epoch:
            current_state_loss_weights = base_state_loss_weights.at[
                OBJECT_INDICES
            ].multiply(args.object_state_weight)
        order = np.asarray(jr.permutation(jr.PRNGKey(args.seed + 10 + epoch), len(train[0])))
        if args.samples_per_epoch > 0:
            order = order[:min(args.samples_per_epoch, len(order))]
        losses, base_grad_norms, contact_grad_norms = [], [], []
        anneal = min(1.0, (epoch + 1) / max(1, args.contact_anneal_epochs))
        contact_coefficient = args.contact_force_supervision_weight * anneal
        trajectory_coefficient = args.object_trajectory_supervision_weight * anneal
        for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start:start + args.batch_size]
            batch = tuple(value[indices] for value in train)
            key = jr.fold_in(jr.PRNGKey(args.seed + 1000 + epoch), batch_number)
            model, opt_state, loss, base_grad_norm, contact_grad_norm = train_step(
                model, opt_state, batch, key, current_state_loss_weights,
                contact_coefficient, trajectory_coefficient,
            )
            losses.append(float(loss))
            base_grad_norms.append(float(base_grad_norm))
            contact_grad_norms.append(float(contact_grad_norm))
        # Use a deterministic fixed validation subset/key for comparable epochs.
        validation_order = np.asarray(jr.permutation(
            jr.PRNGKey(args.seed + 8000), len(validation[0])
        ))[:min(args.validation_windows, len(validation[0]))]
        validation_subset = tuple(value[validation_order] for value in validation)
        validation_loss = float(model.multiple_shooting_loss(
            *validation_subset[:3], jr.PRNGKey(args.seed + 9000),
            segment_length=args.segment_length,
            n_particles=args.train_particles, solver_name=args.solver,
            shooting_noise_std=0.0,
            control_noise_std=0.0,
            path_weight=args.path_weight,
            uncertainty_weight=args.uncertainty_weight,
            continuity_weight=args.continuity_weight,
            state_loss_weights=current_state_loss_weights,
            contact_weight=args.contact_weight,
            contact_center=args.contact_center,
            contact_width=args.contact_width,
        ))
        validation_times = (
            validation_subset[2][:, None]
            + jnp.arange(validation_subset[1].shape[1])[None, :]
        ) * args.dt
        validation_loss += args.transition_weight * float(
            model.transition_nll(
                validation_subset[0][:, :-1], validation_subset[1],
                validation_subset[0][:, 1:], validation_times,
            )
        )
        if args.state_mse_weight != 0:
            validation_x = validation_subset[0][:, :-1].reshape(-1, state_dim)
            validation_u = validation_subset[1].reshape(-1, control_dim)
            validation_target = validation_subset[0][:, 1:].reshape(-1, state_dim)
            validation_drift = jax.vmap(model.drift)(
                validation_x, validation_u, validation_times.reshape(-1)
            )
            validation_prediction = validation_x + args.dt * validation_drift
            validation_loss += args.state_mse_weight * float(jnp.mean(
                current_state_loss_weights
                * (validation_prediction - validation_target) ** 2
            ))
        if len(validation_subset) == 4:
            validation_x = validation_subset[0][:, :-1].reshape(-1, state_dim)
            validation_u = validation_subset[1].reshape(-1, control_dim)
            validation_target = validation_subset[0][:, 1:].reshape(-1, state_dim)
            validation_drift = jax.vmap(model.drift)(
                validation_x, validation_u, validation_times.reshape(-1)
            )
            validation_prediction = validation_x + args.dt * validation_drift
            force = validation_subset[3].reshape(
                -1, validation_subset[3].shape[-1]
            )
            intensity = contact_temporal_weights(force, contact_force_scale)
            object_indices = OBJECT_INDICES
            object_error = jnp.mean(
                (validation_prediction[:, object_indices]
                 - validation_target[:, object_indices]) ** 2, axis=-1
            )
            validation_loss += contact_coefficient * float(
                jnp.sum(intensity * object_error) / jnp.maximum(
                    jnp.sum(intensity), 1.0
                )
            )
            if args.object_trajectory_supervision_weight > 0:
                object_weights = jnp.zeros((state_dim,))
                object_weights = object_weights.at[OBJECT_INDICES].set(1.0)
                validation_loss += trajectory_coefficient * float(
                    model.multiple_shooting_loss(
                        *validation_subset[:3], jr.PRNGKey(args.seed + 9173),
                        segment_length=args.segment_length,
                        n_particles=args.train_particles,
                        solver_name=args.solver,
                        shooting_noise_std=0.0, control_noise_std=0.0,
                        path_weight=1.0,
                        uncertainty_weight=args.uncertainty_weight,
                        continuity_weight=args.continuity_weight,
                        state_loss_weights=object_weights,
                        temporal_loss_weights=intensity.reshape(
                            validation_subset[3].shape[:2]
                        ),
                        contact_weight=1.0,
                    )
                )
        train_loss = float(np.mean(losses))
        validation_x = validation_subset[0][:, :-1].reshape(-1, state_dim)
        validation_u = validation_subset[1].reshape(-1, control_dim)
        validation_target = validation_subset[0][:, 1:].reshape(-1, state_dim)
        validation_prediction = validation_x + args.dt * jax.vmap(model.drift)(
            validation_x, validation_u, validation_times.reshape(-1)
        )
        validation_arm_mse = float(jnp.mean(
            (validation_prediction[:, ARM_INDICES]
             - validation_target[:, ARM_INDICES]) ** 2
        ))
        validation_object_mse = float(jnp.mean(
            (validation_prediction[:, OBJECT_INDICES]
             - validation_target[:, OBJECT_INDICES]) ** 2
        ))
        base_grad_mean = float(np.mean(base_grad_norms))
        contact_grad_mean = float(np.mean(contact_grad_norms))
        curve.append((epoch + 1, train_loss, validation_loss,
                      validation_arm_mse, validation_object_mse,
                      base_grad_mean, contact_grad_mean, contact_coefficient,
                      trajectory_coefficient))
        if validation_loss < best_validation:
            best_model, best_validation = model, validation_loss
        print(f"epoch={epoch + 1:03d} train_path_loss={train_loss:.6f} "
              f"validation_path_loss={validation_loss:.6f} "
              f"val_arm_mse={validation_arm_mse:.6f} "
              f"val_object_mse={validation_object_mse:.6f}", flush=True)
        if wandb_run is not None:
            active_lr = (
                args.lr_after_decay
                if epoch + 1 > args.lr_decay_epoch else args.lr
            )
            active_object_weight = (
                args.object_state_weight
                if epoch + 1 > args.state_weight_epoch else 1.0
            )
            wandb_run.log({
                "epoch": epoch + 1,
                "loss/train": train_loss,
                "loss/validation": validation_loss,
                "loss/best_validation": best_validation,
                "optimizer/learning_rate": active_lr,
                "loss/object_state_weight": active_object_weight,
                "loss/validation_arm_mse": validation_arm_mse,
                "loss/validation_object_mse": validation_object_mse,
                "gradient/base_norm": base_grad_mean,
                "gradient/contact_norm": contact_grad_mean,
                "loss/contact_coefficient": contact_coefficient,
                "loss/object_trajectory_coefficient": trajectory_coefficient,
            }, step=epoch + 1)

    if args.state_is_position_velocity:
        diffusion_tag = "_qv_existing"
    elif args.velocity_augmented:
        diffusion_tag = "_qv"
    else:
        diffusion_tag = "_gx" if args.diffusion_no_control else ""
    tag = (f"aloha_flow_sde_{args.solver}_seg{args.segment_length}{diffusion_tag}_"
           f"noise{str(args.state_noise_std).replace('.', 'p')}_seed{args.seed}")
    checkpoint = args.output / f"{tag}.eqx"
    normalization_path = args.output / f"{tag}_normalization.npz"
    eqx.tree_serialise_leaves(checkpoint, best_model)
    np.savez(normalization_path, x_mean=np.asarray(stats[0]), x_std=np.asarray(stats[1]),
             u_mean=np.asarray(stats[2]), u_std=np.asarray(stats[3]))
    with (args.output / f"{tag}_learning_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow([
            "epoch", "train_path_loss", "validation_path_loss",
            "validation_arm_mse", "validation_object_mse",
            "base_gradient_norm", "contact_gradient_norm", "contact_coefficient",
            "object_trajectory_coefficient",
        ])
        writer.writerows(curve)
    config = vars(args).copy()
    config.update({"data": str(args.data), "output": str(args.output),
                   "checkpoint": str(checkpoint), "normalization": str(normalization_path),
                   "state_dim": state_dim, "control_dim": control_dim,
                   "position_dim": position_dim,
                   "training_window_pool": len(train[0]),
                   "validation_window_pool": len(validation[0]),
                   "effective_training_windows_per_epoch": (
                       min(args.samples_per_epoch, len(train[0]))
                       if args.samples_per_epoch > 0 else len(train[0])
                   ),
                   "train_episode_ids": episode_ids[train_i].tolist(),
                   "validation_episode_ids": episode_ids[val_i].tolist(),
                   "test_episode_ids": episode_ids[test_i].tolist(),
                   "training_objective": (
                       "transition_nll_plus_multiple_shooting_continuity"
                       if args.path_weight == 0 and args.transition_weight > 0
                       else "weighted_transition_and_stochastic_multiple_shooting"
                   ),
                   "state_loss_weights": np.asarray(state_loss_weights).tolist(),
                   "contact_force_scale": (
                       None if contact_force_scale is None
                       else contact_force_scale.tolist()
                   ),
                   "best_validation_path_loss": best_validation,
                   "test_transition_nll": float(best_model.transition_nll(*test)),
                   "elapsed_seconds": time.perf_counter() - started})
    with (args.output / f"{tag}_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    if wandb_run is not None:
        wandb_run.summary["best_validation_loss"] = best_validation
        wandb_run.summary["test_transition_nll"] = config["test_transition_nll"]
        wandb_run.summary["elapsed_seconds"] = config["elapsed_seconds"]
        wandb_run.finish()
    print(json.dumps(config, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/aloha_mobile_cabinet/flow_sde"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-epoch", type=int, default=200)
    p.add_argument("--lr-after-decay", type=float, default=3e-4)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--diffusion-width", type=int, default=64)
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--time-embedding", type=int, default=16)
    p.add_argument("--realnvp-alpha", type=float, default=3.0)
    p.add_argument("--diffusion-no-control", action="store_true",
                   help="Condition g on (epsilon, x, t), while f still uses (x, u, t).")
    p.add_argument("--object-drift-no-control", action="store_true",
                   help="Remove direct control from q4:6/v4:6 drift outputs.")
    p.add_argument("--separate-dynamics-heads", action="store_true",
                   help="Independent drift and diffusion heads for arm/object groups.")
    p.add_argument("--use-contact-geometry", action="store_true",
                   help="Condition dynamics on inference-safe pusher/object geometry.")
    p.add_argument("--geometry-temperature", type=float, default=0.02)
    p.add_argument("--velocity-augmented", action="store_true",
                   help="Use structured state [q,v], dq=v and stochastic dv.")
    p.add_argument("--state-is-position-velocity", action="store_true",
                   help="Use an existing [q,v] state with structured second-order dynamics.")
    p.add_argument("--velocity-window", type=int, default=3,
                   help="Backward causal finite-difference window in frames.")
    p.add_argument("--state-noise-std", type=float, default=0.01,
                   help="Gaussian perturbation of multiple-shooting initial states.")
    p.add_argument("--control-noise-std", type=float, default=0.01,
                   help="Gaussian perturbation on normalized control windows.")
    p.add_argument("--window-steps", type=int, default=128)
    p.add_argument("--segment-length", type=int, default=8)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=0,
                   help="Random windows drawn from the dense pool each epoch; 0 uses all.")
    p.add_argument("--train-particles", type=int, default=4)
    p.add_argument("--solver", choices=("euler", "heun", "reversible_heun",
                   "midpoint", "milstein"), default="heun")
    p.add_argument("--uncertainty-weight", type=float, default=0.5)
    p.add_argument("--path-weight", type=float, default=1.0,
                   help="Weight of the pathwise energy-score term.")
    p.add_argument("--transition-weight", type=float, default=1.0,
                   help="Weight of exact one-step RealNVP transition NLL.")
    p.add_argument("--state-mse-weight", type=float, default=0.0,
                   help="Weight of auxiliary state-weighted drift one-step MSE.")
    p.add_argument("--state-loss-weights", type=str, default="",
                   help="Comma-separated normalized-space weights per state dimension.")
    p.add_argument("--state-weight-epoch", type=int, default=200)
    p.add_argument("--object-state-weight", type=float, default=3.0,
                   help="Multiplier for q4:6/v4:6 starting after state-weight-epoch.")
    p.add_argument("--continuity-weight", type=float, default=0.1)
    p.add_argument("--contact-weight", type=float, default=3.0,
                   help="Peak temporal weight around the contact phase.")
    p.add_argument("--contact-center", type=float, default=0.5,
                   help="Contact-phase center as a fraction of the trajectory.")
    p.add_argument("--contact-width", type=float, default=0.15,
                   help="Gaussian contact-phase width as a trajectory fraction.")
    p.add_argument("--contact-force-supervision-weight", type=float, default=0.0,
                   help="Training-only q4:6/v4:6 MSE weight gated by MuJoCo contact force.")
    p.add_argument("--contact-force-column", default="object_contact_force",
                   help="Parquet column containing per-transition [Fx,Fy,tau_z].")
    p.add_argument("--object-trajectory-supervision-weight", type=float, default=0.0,
                   help="Training-only stochastic multi-step object trajectory energy score.")
    p.add_argument("--contact-anneal-epochs", type=int, default=25,
                   help="Epochs used to linearly ramp contact supervision from zero.")
    p.add_argument("--validation-windows", type=int, default=8)
    p.add_argument("--wandb-project", type=str, default="",
                   help="W&B project name; empty disables W&B logging.")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
