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

from bayesinf.applications.optimal_control.controlled_flow_sde import ControlledRealNVPFlowSDE
from bayesinf.applications.optimal_control.second_order_flow_sde import SecondOrderControlledRealNVPFlowSDE
from .data import (
    augment_with_causal_velocity, load_episodes, load_transition_auxiliary,
    normalization,
    split_episode_indices,
)

ARM_INDICES = jnp.asarray([0, 1, 2, 6, 7, 8])
OBJECT_INDICES = jnp.asarray([3, 4, 5, 9, 10, 11])


def contact_geometry_feature_fn(enabled):
    """Load optional prototype geometry only when the feature is requested."""
    if not enabled:
        return None
    try:
        from experiments.mujoco_pusher_optimal_control.geometry import pusher_contact_features
    except ModuleNotFoundError as error:
        if error.name not in {
            "experiments.mujoco_pusher_optimal_control",
            "experiments.mujoco_pusher_optimal_control.geometry",
        }:
            raise
        raise RuntimeError(
            "--use-contact-geometry requires the optional pusher geometry prototype, "
            "which is not included in this public snapshot; run without that option"
        ) from error
    return pusher_contact_features


def contact_temporal_weights(forces, scale, post_contact_steps=0):
    """Smooth positive trajectory weights derived from training-only wrench labels."""
    raw = jnp.sqrt(jnp.mean((forces / jnp.asarray(scale)) ** 2, axis=-1))
    if post_contact_steps > 0:
        if raw.ndim != 2:
            raise ValueError("post-contact weighting requires [batch, time, wrench]")
        raw = jax.lax.reduce_window(
            raw, -jnp.inf, jax.lax.max,
            window_dimensions=(1, post_contact_steps + 1),
            window_strides=(1, 1),
            padding=((0, 0), (post_contact_steps, 0)),
        )
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
        diffusion_condition_on_geometry=not getattr(
            args, "diffusion_no_geometry", False
        ),
        object_drift_no_control=getattr(args, "object_drift_no_control", False),
        separate_dynamics_heads=getattr(args, "separate_dynamics_heads", False),
        coupled_six_state_flows=getattr(args, "coupled_six_state_flows", False),
        use_contact_geometry=getattr(args, "use_contact_geometry", False),
        object_history_steps=getattr(args, "object_history_steps", 0),
        kinematic_position_drift=getattr(args, "kinematic_position_drift", False),
        geometry_feature_dim=getattr(args, "geometry_feature_dim", 5),
        contact_gated_object_drift=getattr(
            args, "contact_gated_object_drift", False
        ),
        geometry_feature_fn=contact_geometry_feature_fn(
            getattr(args, "use_contact_geometry", False)
        ),
        geometry_temperature=getattr(args, "geometry_temperature", 0.02),
        state_mean=getattr(args, "model_state_mean", None),
        state_std=getattr(args, "model_state_std", None),
        geometry_mean=getattr(args, "geometry_mean", None),
        geometry_std=getattr(args, "geometry_std", None),
    )


def run(args):
    geometry_feature_fn = contact_geometry_feature_fn(getattr(args, "use_contact_geometry", False))
    args.output.mkdir(parents=True, exist_ok=True)
    if not hasattr(args, "object_trajectory_supervision_weight"):
        args.object_trajectory_supervision_weight = 0.0
    if not hasattr(args, "object_segment_length"):
        args.object_segment_length = args.segment_length
    if not hasattr(args, "post_contact_steps"):
        args.post_contact_steps = 0
    if not hasattr(args, "initial_train_particles"):
        args.initial_train_particles = args.train_particles
    if not hasattr(args, "particle_finetune_epoch"):
        args.particle_finetune_epoch = args.epochs
    if not hasattr(args, "validation_interval"):
        args.validation_interval = 1
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
    if args.object_trajectory_supervision_weight > 0 and args.object_segment_length <= 0:
        raise ValueError("--object-segment-length must be positive")
    if (args.object_trajectory_supervision_weight > 0
            and args.window_steps % args.object_segment_length):
        raise ValueError("--window-steps must be divisible by --object-segment-length")
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
            lambda state: geometry_feature_fn(
                state, temperature=args.geometry_temperature
            )[:args.geometry_feature_dim]
        )(jnp.asarray(train_states)))
        args.geometry_mean = geometry.mean(0).tolist()
        args.geometry_std = np.maximum(geometry.std(0), 1e-4).tolist()
    else:
        args.geometry_mean = [0.0] * args.geometry_feature_dim
        args.geometry_std = [1.0] * args.geometry_feature_dim
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

    # Keep every state in the deployable path objective. Contact labels only
    # change temporal emphasis; they must not remove object states from the
    # ordinary trajectory loss.
    base_state_loss_weights = state_loss_weights

    def make_train_step(n_particles):
        @eqx.filter_jit
        def train_step(current_model, current_state, batch, noise_key,
                       current_state_loss_weights, contact_coefficient,
                       trajectory_coefficient):
            bx, bu, bstart = batch[:3]
            bcontact = batch[3] if len(batch) == 4 else None

            def objective(m):
                intensity = (
                    jnp.ones(bu.shape[:2], dtype=bx.dtype)
                    if bcontact is None else contact_temporal_weights(
                        bcontact, contact_force_scale, args.post_contact_steps
                    )
                )
                if args.coupled_six_state_flows:
                    robot_path, object_path = m.coupled_multiple_shooting_losses(
                        bx, bu, bstart, noise_key,
                        segment_length=args.object_segment_length,
                        n_particles=n_particles, solver_name=args.solver,
                        shooting_noise_std=args.state_noise_std,
                        control_noise_std=args.control_noise_std,
                        path_weight=args.path_weight,
                        object_path_weight=1.0,
                        uncertainty_weight=args.uncertainty_weight,
                        continuity_weight=args.continuity_weight,
                        object_temporal_weights=intensity,
                    )
                else:
                    robot_path = m.multiple_shooting_loss(
                        bx, bu, bstart, noise_key,
                        segment_length=args.segment_length,
                        n_particles=n_particles, solver_name=args.solver,
                        shooting_noise_std=args.state_noise_std,
                        control_noise_std=args.control_noise_std,
                        path_weight=args.path_weight,
                        uncertainty_weight=args.uncertainty_weight,
                        continuity_weight=args.continuity_weight,
                        state_loss_weights=current_state_loss_weights,
                        temporal_loss_weights=intensity,
                    )
                    object_path = jnp.asarray(0.0)
                times = (
                    bstart[:, None] + jnp.arange(bu.shape[1])[None, :]
                ) * args.dt
                transition = m.noisy_transition_nll(
                    bx[:, :-1], bu, bx[:, 1:], times,
                    jr.fold_in(noise_key, 71),
                    state_noise_std=args.state_noise_std,
                    control_noise_std=args.control_noise_std,
                    state_history=m.sequence_histories(bx),
                )
                x_flat = bx[:, :-1].reshape(-1, state_dim)
                u_flat = bu.reshape(-1, control_dim)
                target_flat = bx[:, 1:].reshape(-1, state_dim)
                histories = m.sequence_histories(bx)
                if histories is None:
                    drift = jax.vmap(m.drift)(
                        x_flat, u_flat, times.reshape(-1)
                    )
                else:
                    drift = jax.vmap(m.drift)(
                        x_flat, u_flat, times.reshape(-1),
                        histories.reshape(-1, m.object_history_steps, state_dim),
                    )
                predicted = x_flat + args.dt * drift
                weighted_mse = jnp.mean(
                    current_state_loss_weights * (predicted - target_flat) ** 2
                )
                object_error = jnp.mean(
                    current_state_loss_weights[OBJECT_INDICES]
                    * (predicted[:, OBJECT_INDICES]
                       - target_flat[:, OBJECT_INDICES]) ** 2, axis=-1
                )
                intensity_flat = intensity.reshape(-1)
                contact_loss = jnp.sum(
                    intensity_flat * object_error
                ) / jnp.maximum(jnp.sum(intensity_flat), 1.0)
                total = (
                    robot_path
                    + trajectory_coefficient * object_path
                    + args.transition_weight * transition
                    + args.state_mse_weight * weighted_mse
                    + contact_coefficient * contact_loss
                )
                diagnostics = (robot_path, object_path, transition, contact_loss)
                return total, diagnostics

            (loss, diagnostics), grads = eqx.filter_value_and_grad(
                objective, has_aux=True
            )(current_model)
            gradient_norm = optax.global_norm(grads)
            updates, next_state = optimizer.update(
                grads, current_state,
                eqx.filter(current_model, eqx.is_inexact_array),
            )
            return (eqx.apply_updates(current_model, updates), next_state,
                    loss, gradient_norm, diagnostics)

        return train_step

    initial_train_step = make_train_step(args.initial_train_particles)
    final_train_step = make_train_step(args.train_particles)

    validation_order = np.asarray(jr.permutation(
        jr.PRNGKey(args.seed + 8000), len(validation[0])
    ))[:min(args.validation_windows, len(validation[0]))]
    validation_subset = tuple(value[validation_order] for value in validation)

    @eqx.filter_jit
    def validation_metrics(current_model, contact_coefficient,
                           trajectory_coefficient):
        vx, vu, vstart = validation_subset[:3]
        vcontact = validation_subset[3] if len(validation_subset) == 4 else None
        intensity = (
            jnp.ones(vu.shape[:2], dtype=vx.dtype)
            if vcontact is None else contact_temporal_weights(
                vcontact, contact_force_scale, args.post_contact_steps
            )
        )
        if args.coupled_six_state_flows:
            robot_path, object_path = current_model.coupled_multiple_shooting_losses(
                vx, vu, vstart, jr.PRNGKey(args.seed + 9000),
                segment_length=args.object_segment_length,
                n_particles=args.train_particles, solver_name=args.solver,
                shooting_noise_std=0.0, control_noise_std=0.0,
                path_weight=args.path_weight,
                object_path_weight=1.0,
                uncertainty_weight=args.uncertainty_weight,
                continuity_weight=args.continuity_weight,
                object_temporal_weights=intensity,
            )
        else:
            robot_path = current_model.multiple_shooting_loss(
                vx, vu, vstart, jr.PRNGKey(args.seed + 9000),
                segment_length=args.segment_length,
                n_particles=args.train_particles, solver_name=args.solver,
                shooting_noise_std=0.0, control_noise_std=0.0,
                path_weight=args.path_weight,
                uncertainty_weight=args.uncertainty_weight,
                continuity_weight=args.continuity_weight,
                state_loss_weights=base_state_loss_weights,
                temporal_loss_weights=intensity,
            )
            object_path = jnp.asarray(0.0)
        times = (vstart[:, None] + jnp.arange(vu.shape[1])[None, :]) * args.dt
        transition = current_model.transition_nll(
            vx[:, :-1], vu, vx[:, 1:], times,
            current_model.sequence_histories(vx),
        )
        flat_x = vx[:, :-1].reshape(-1, state_dim)
        flat_u = vu.reshape(-1, control_dim)
        target = vx[:, 1:].reshape(-1, state_dim)
        histories = current_model.sequence_histories(vx)
        if histories is None:
            drift = jax.vmap(current_model.drift)(
                flat_x, flat_u, times.reshape(-1)
            )
        else:
            drift = jax.vmap(current_model.drift)(
                flat_x, flat_u, times.reshape(-1),
                histories.reshape(
                    -1, current_model.object_history_steps, state_dim
                ),
            )
        prediction = flat_x + args.dt * drift
        weighted_mse = jnp.mean(
            base_state_loss_weights * (prediction - target) ** 2
        )
        object_error = jnp.mean(
            base_state_loss_weights[OBJECT_INDICES]
            * (prediction[:, OBJECT_INDICES] - target[:, OBJECT_INDICES]) ** 2,
            axis=-1,
        )
        contact_loss = jnp.sum(
            intensity.reshape(-1) * object_error
        ) / jnp.maximum(jnp.sum(intensity), 1.0)
        total = (
            robot_path + trajectory_coefficient * object_path
            + args.transition_weight * transition
            + args.state_mse_weight * weighted_mse
            + contact_coefficient * contact_loss
        )
        arm_mse = jnp.mean(
            (prediction[:, ARM_INDICES] - target[:, ARM_INDICES]) ** 2
        )
        object_mse = jnp.mean(
            (prediction[:, OBJECT_INDICES] - target[:, OBJECT_INDICES]) ** 2
        )
        return total, robot_path, object_path, arm_mse, object_mse

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
        losses, gradient_norms, robot_path_losses, object_path_losses = [], [], [], []
        anneal = min(1.0, (epoch + 1) / max(1, args.contact_anneal_epochs))
        contact_coefficient = args.contact_force_supervision_weight * anneal
        trajectory_coefficient = args.object_trajectory_supervision_weight * anneal
        train_step = (
            final_train_step if epoch >= args.particle_finetune_epoch
            else initial_train_step
        )
        active_particles = (
            args.train_particles if epoch >= args.particle_finetune_epoch
            else args.initial_train_particles
        )
        for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start:start + args.batch_size]
            batch = tuple(value[indices] for value in train)
            key = jr.fold_in(jr.PRNGKey(args.seed + 1000 + epoch), batch_number)
            model, opt_state, loss, gradient_norm, diagnostics = train_step(
                model, opt_state, batch, key, current_state_loss_weights,
                contact_coefficient, trajectory_coefficient,
            )
            losses.append(float(loss))
            gradient_norms.append(float(gradient_norm))
            robot_path_losses.append(float(diagnostics[0]))
            object_path_losses.append(float(diagnostics[1]))
        train_loss = float(np.mean(losses))
        gradient_norm_mean = float(np.mean(gradient_norms))
        train_robot_path = float(np.mean(robot_path_losses))
        train_object_path = float(np.mean(object_path_losses))
        validate_now = (
            epoch == 0 or (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == args.epochs
        )
        if validate_now:
            metrics = validation_metrics(
                model, contact_coefficient, trajectory_coefficient
            )
            (validation_loss, validation_robot_path, validation_object_path,
             validation_arm_mse, validation_object_mse) = map(float, metrics)
            if validation_loss < best_validation:
                best_model, best_validation = model, validation_loss
        else:
            (validation_loss, validation_robot_path, validation_object_path,
             validation_arm_mse, validation_object_mse) = (float("nan"),) * 5
        curve.append((epoch + 1, train_loss, validation_loss,
                      validation_arm_mse, validation_object_mse,
                      train_robot_path, train_object_path,
                      validation_robot_path, validation_object_path,
                      gradient_norm_mean, active_particles,
                      contact_coefficient, trajectory_coefficient))
        message = (
            f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
            f"robot_path={train_robot_path:.6f} "
            f"object_path={train_object_path:.6f} particles={active_particles}"
        )
        if validate_now:
            message += (
                f" validation={validation_loss:.6f} "
                f"val_robot_mse={validation_arm_mse:.6f} "
                f"val_object_mse={validation_object_mse:.6f}"
            )
        print(message, flush=True)
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
                "loss/train_robot_path": train_robot_path,
                "loss/train_object_path": train_object_path,
                "loss/validation_robot_path": validation_robot_path,
                "loss/validation_object_path": validation_object_path,
                "gradient/global_norm": gradient_norm_mean,
                "optimizer/train_particles": active_particles,
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
            "epoch", "train_loss", "validation_loss",
            "validation_arm_mse", "validation_object_mse",
            "train_robot_path_loss", "train_object_path_loss",
            "validation_robot_path_loss", "validation_object_path_loss",
            "gradient_norm", "train_particles", "contact_coefficient",
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
    p.add_argument("--diffusion-no-geometry", action="store_true",
                   help="Keep contact geometry in drift but exclude it from diffusion.")
    p.add_argument("--object-drift-no-control", action="store_true",
                   help="Remove direct control from q4:6/v4:6 drift outputs.")
    p.add_argument("--separate-dynamics-heads", action="store_true",
                   help="Independent drift and diffusion heads for arm/object groups.")
    p.add_argument("--coupled-six-state-flows", action="store_true",
                   help="Jointly train autoregressively coupled 6-D robot/object flows.")
    p.add_argument("--use-contact-geometry", action="store_true",
                   help="Condition dynamics on inference-safe pusher/object geometry.")
    p.add_argument("--geometry-temperature", type=float, default=0.02)
    p.add_argument("--geometry-feature-dim", type=int, choices=(5, 9), default=5,
                   help="Use basic 5-D or torque-aware extended 9-D geometry.")
    p.add_argument("--contact-gated-object-drift", action="store_true",
                   help="Blend free/contact object drift experts using proximity.")
    p.add_argument("--object-history-steps", type=int, default=0,
                   help="Prior full states conditioning only the object drift/flow head.")
    p.add_argument("--kinematic-position-drift", action="store_true",
                   help="Enforce normalized dq/dt=v for 12-D [q,v] pusher states.")
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
    p.add_argument("--initial-train-particles", type=int, default=2,
                   help="Particles used before particle-finetune-epoch.")
    p.add_argument("--particle-finetune-epoch", type=int, default=200,
                   help="Epoch index at which train-particles replaces the initial count.")
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
    p.add_argument("--contact-weight", type=float, default=1.0,
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
    p.add_argument("--object-segment-length", type=int, default=25,
                   help="Multi-step horizon used by the object trajectory objective.")
    p.add_argument("--post-contact-steps", type=int, default=10,
                   help="Keep contact weighting active for this many subsequent steps.")
    p.add_argument("--contact-anneal-epochs", type=int, default=25,
                   help="Epochs used to linearly ramp contact supervision from zero.")
    p.add_argument("--validation-windows", type=int, default=8)
    p.add_argument("--validation-interval", type=int, default=5,
                   help="Run stochastic validation every N epochs.")
    p.add_argument("--wandb-project", type=str, default="",
                   help="W&B project name; empty disables W&B logging.")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
