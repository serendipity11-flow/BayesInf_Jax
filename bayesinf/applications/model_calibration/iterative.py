"""JAX implementation of iterative forward/inverse thermal calibration."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.models.normalizing_flow import BNNormFlow


DEFAULT_IDENTIFICATION_GROUPS = (
    (11, 14, 15, 16),
    (12, 13, 17),
    (6, 7, 8, 10, 18),
    (0, 1, 2, 3, 4),
    (5, 9),
)

# Physical sampling bounds used to generate the FMU calibration data.  Keeping
# these fixed across stages makes the inverse-model coordinates independent of
# the random train/validation split.
THERMAL_PARAMETER_BOUNDS = np.asarray((
    (18.0, 27.0), (35.0, 55.0), (0.05, 1.0), (0.5, 1.5), (1.0, 2.5),
    (0.5, 1.5), (0.05, 1.0), (0.05, 1.0), (0.05, 1.0), (0.5, 1.5),
    (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.05, 1.0), (0.05, 1.0),
    (0.05, 1.0), (0.5, 1.5), (20.0, 35.0), (2.0, 3.5),
), dtype=np.float32)


class ForwardModel(Protocol):
    def __call__(self, parameters: np.ndarray, key: jax.Array) -> np.ndarray: ...


@eqx.filter_jit
def _stateful_flow_step(model, state, signals, parameters, optimizer_state, optimizer):
    def objective(candidate, current_state):
        log_probability, updated_state = candidate.log_prob_with_state(
            parameters, signals, current_state, inference=False
        )
        return -jnp.mean(log_probability), updated_state
    (loss, state), gradients = eqx.filter_value_and_grad(
        objective, has_aux=True
    )(model, state)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    return eqx.apply_updates(model, updates), state, optimizer_state, loss


@eqx.filter_jit
def _stateful_flow_validation(model, state, signals, parameters):
    log_probability, _ = model.log_prob_with_state(
        parameters, signals, state, inference=True
    )
    return -jnp.mean(log_probability)


@dataclass(frozen=True)
class CalibrationConfig:
    groups: tuple[tuple[int, ...], ...] = DEFAULT_IDENTIFICATION_GROUPS
    epochs: int = 1000
    batch_size: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 1_000
    min_delta: float = 1e-5
    posterior_samples: int = 100
    n_coupling: int = 4
    width: int = 64
    summary_dim: int = 64
    rnn_hidden: int = 64
    recurrent_layers: int = 2
    clamp: float = 1.0
    lr_hold_epochs: int = 500
    min_learning_rate: float = 5e-6
    intermediate_eval_every: int = 100
    intermediate_posterior_samples: int = 100
    seed: int = 0


@dataclass(frozen=True)
class CalibrationResult:
    estimate: np.ndarray
    truth: np.ndarray
    posterior_samples: tuple[np.ndarray, ...]
    groups: tuple[tuple[int, ...], ...]

    @property
    def relative_error_percent(self) -> np.ndarray:
        scale = np.maximum(np.abs(self.truth), 1e-8)
        return np.abs(self.estimate - self.truth) / scale * 100.0


class PELSVAEForwardModel:
    """Batch adapter for the existing Equinox :class:`PELS_VAE` surrogate."""

    def __init__(self, model, state, normalization, *, passes: int = 1):
        self.model = eqx.nn.inference_mode(model)
        self.state = state
        self.normalization = normalization
        self.passes = passes

    def __call__(self, parameters: np.ndarray, key: jax.Array) -> np.ndarray:
        theta = jnp.asarray(self.normalization.normalize_parameters(parameters))
        prediction = np.asarray(self.model.predict(
            theta, self.state, key=key, passes=self.passes), dtype=np.float32)
        return self.normalization.denormalize_signals(prediction)


def _global_minmax(train: np.ndarray, *others: np.ndarray):
    """Map signals to [-1, 1] using one training-only global range."""
    minimum = np.asarray(train.min(), dtype=np.float32)
    maximum = np.asarray(train.max(), dtype=np.float32)
    scale = np.maximum(maximum - minimum, np.float32(1e-6))
    transform = lambda value: 2.0 * (value - minimum) / scale - 1.0
    return (transform(train), *(transform(value) for value in others), minimum, maximum)


def _parameter_minmax(parameters: np.ndarray, indices: Sequence[int]):
    """Map selected physical parameters to [-1, 1] dimension by dimension."""
    del indices
    minimum = parameters.min(axis=0, keepdims=True)
    maximum = parameters.max(axis=0, keepdims=True)
    normalized = 2.0 * (parameters - minimum) / (maximum - minimum) - 1.0
    return normalized, minimum, maximum


def _parameter_inverse(normalized: np.ndarray, minimum: np.ndarray,
                       maximum: np.ndarray):
    return 0.5 * (normalized + 1.0) * (maximum - minimum) + minimum


class IterativeModelCalibrator:
    """Identify sensitivity-ordered groups and regenerate training signals."""

    def __init__(self, config: CalibrationConfig, output_dir: Path | str, tracker=None,
                 tracker_loss_only: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.tracker = tracker
        self.tracker_loss_only = tracker_loss_only
        self._tracking_step = 0

    def _fit_group(self, train_x, train_theta, val_x, val_theta, group, key, stage_dir,
                   normalized_target, theta_min, theta_max, truth_group, stage):
        model_key, loop_key = jr.split(key)
        model = BNNormFlow(
            len(group), train_x.shape[-1], model_key,
            n_coupling=self.config.n_coupling,
            hidden_dims=(self.config.width,) * 4,
            encoded_dim=self.config.summary_dim,
            encoder_hidden_size=self.config.rnn_hidden,
            encoder_layers=self.config.recurrent_layers,
            sequence_length=train_x.shape[1],
            activation=jnp.tanh,
            clamp=self.config.clamp,
            clamp_activation="I",
        )
        flow_state = model.initial_batchnorm_state()
        model = model.initialize_actnorm(
            jnp.asarray(train_theta), jnp.asarray(train_x), flow_state
        )
        steps_per_epoch = max(1, (len(train_x) + self.config.batch_size - 1) // self.config.batch_size)
        def learning_rate(step):
            epoch = jnp.asarray(step, dtype=jnp.float32) / steps_per_epoch
            decay_epochs = max(1, self.config.epochs - self.config.lr_hold_epochs)
            progress = jnp.clip(
                (epoch - self.config.lr_hold_epochs) / decay_epochs, 0.0, 1.0
            )
            cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            return self.config.min_learning_rate + (
                self.config.learning_rate - self.config.min_learning_rate
            ) * cosine
        optimizer = optax.chain(
            optax.add_decayed_weights(self.config.weight_decay),
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(learning_rate, flip_sign=True),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        best, best_state, best_loss, stale, history = model, flow_state, np.inf, 0, []
        intermediate_path = stage_dir / "intermediate_metrics.csv"
        with intermediate_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("epoch", "validation_loss", "posterior_rmse", "posterior_mae",
                             *(f"mean_parameter_{index}" for index in group)))
        x, y = jnp.asarray(train_x), jnp.asarray(train_theta)
        vx, vy = jnp.asarray(val_x), jnp.asarray(val_theta)
        training_start = time.perf_counter()
        for epoch in range(self.config.epochs):
            loop_key, permutation_key, step_key = jr.split(loop_key, 3)
            permutation = np.asarray(jr.permutation(permutation_key, len(x)))
            losses = []
            for begin in range(0, len(x), self.config.batch_size):
                ids = permutation[begin:begin + self.config.batch_size]
                model, flow_state, opt_state, loss = _stateful_flow_step(
                    model, flow_state, x[ids], y[ids], opt_state, optimizer
                )
                losses.append(float(loss))
            val_loss = float(_stateful_flow_validation(model, flow_state, vx, vy))
            history.append((epoch + 1, float(np.mean(losses)), val_loss))
            if np.isfinite(val_loss) and val_loss < best_loss - self.config.min_delta:
                best, best_state, best_loss, stale = model, flow_state, val_loss, 0
                eqx.tree_serialise_leaves(stage_dir / "bayesflow.eqx", best)
                eqx.tree_serialise_leaves(stage_dir / "bayesflow_state.eqx", best_state)
            else:
                stale += 1
            self._tracking_step += 1
            if epoch + 1 <= self.config.lr_hold_epochs:
                current_lr = self.config.learning_rate
            else:
                progress = min(1.0, (epoch + 1 - self.config.lr_hold_epochs) /
                               max(1, self.config.epochs - self.config.lr_hold_epochs))
                current_lr = self.config.min_learning_rate + (
                    self.config.learning_rate - self.config.min_learning_rate
                ) * 0.5 * (1.0 + np.cos(np.pi * progress))
            if self.tracker is not None:
                payload = {"train/loss": history[-1][1]}
                if not self.tracker_loss_only:
                    payload.update({
                        "workflow_step": self._tracking_step,
                        "stage": stage + 1,
                        "stage_epoch": epoch + 1,
                        "validation/loss": val_loss,
                        "optimization/learning_rate": current_lr,
                        "optimization/stale_epochs": stale,
                    })
                self.tracker.log(payload, step=self._tracking_step)
            check_now = epoch == 0 or (epoch + 1) % self.config.intermediate_eval_every == 0
            if check_now:
                step_key, check_key = jr.split(step_key)
                normalized_samples = np.asarray(model.sample(
                    jnp.asarray(normalized_target), check_key,
                    self.config.intermediate_posterior_samples, flow_state,
                ))[:, 0]
                physical_samples = _parameter_inverse(
                    normalized_samples, theta_min, theta_max
                )
                posterior_mean = physical_samples.mean(axis=0)
                posterior_rmse = float(np.sqrt(np.mean((posterior_mean - truth_group) ** 2)))
                posterior_mae = float(np.mean(np.abs(posterior_mean - truth_group)))
                np.save(stage_dir / "latest_intermediate_posterior.npy",
                        physical_samples.astype(np.float32))
                intermediate = {
                    "epoch": epoch + 1, "validation_loss": val_loss,
                    "posterior_rmse": posterior_rmse, "posterior_mae": posterior_mae,
                    "posterior_mean": posterior_mean.tolist(), "group": list(group),
                }
                (stage_dir / "latest_intermediate_metrics.json").write_text(
                    json.dumps(intermediate, indent=2)
                )
                with intermediate_path.open("a", newline="") as handle:
                    csv.writer(handle).writerow((epoch + 1, val_loss, posterior_rmse,
                                                 posterior_mae, *posterior_mean))
                if self.tracker is not None and not self.tracker_loss_only:
                    tracked = {
                        "workflow_step": self._tracking_step,
                        "intermediate/posterior_rmse": posterior_rmse,
                        "intermediate/posterior_mae": posterior_mae,
                    }
                    tracked.update({f"intermediate/parameter_{index}_mean": float(value)
                                    for index, value in zip(group, posterior_mean)})
                    self.tracker.log(tracked, step=self._tracking_step)
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(f"stage={stage + 1} epoch={epoch + 1} train={history[-1][1]:.6f} "
                      f"validation={val_loss:.6f} stale={stale}", flush=True)
            if stale >= self.config.patience:
                break
        training_seconds = time.perf_counter() - training_start
        # Covers the unlikely case where all validation losses are non-finite.
        if not (stage_dir / "bayesflow.eqx").exists():
            eqx.tree_serialise_leaves(stage_dir / "bayesflow.eqx", best)
            eqx.tree_serialise_leaves(stage_dir / "bayesflow_state.eqx", best_state)
        with (stage_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("epoch", "train_loss", "validation_loss"))
            writer.writerows(history)
        return best, best_state, best_loss, len(history), training_seconds

    def run(self, data, forward_model: ForwardModel, *, real_test_index: int = 0,
            max_stages: int | None = None) -> CalibrationResult:
        """Run all groups for one noisy proxy-real observation trajectory."""
        if not 0 <= real_test_index < len(data.real_test.signals):
            raise IndexError("real_test_index is outside the real-test partition")
        parameter_dim = data.train.parameters.shape[1]
        flattened = [index for group in self.config.groups for index in group]
        if sorted(flattened) != list(range(parameter_dim)):
            raise ValueError("groups must contain every parameter exactly once")
        stage_count = len(self.config.groups) if max_stages is None else max_stages
        if not 1 <= stage_count <= len(self.config.groups):
            raise ValueError("max_stages must be between one and the number of groups")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        key = jr.PRNGKey(self.config.seed)
        fixed: dict[int, float] = {}
        posteriors = []
        timings = []
        workflow_start = time.perf_counter()
        target = data.real_test.signals[real_test_index:real_test_index + 1]
        train_x, val_x = data.train.signals, data.validation.signals

        for stage, group in enumerate(self.config.groups[:stage_count]):
            stage_start = time.perf_counter()
            stage_dir = self.output_dir / f"stage_{stage + 1:02d}"
            stage_dir.mkdir(exist_ok=True)
            key, generation_key, training_key, sampling_key = jr.split(key, 4)
            generation_start = time.perf_counter()
            if stage:
                train_parameters = data.train.parameters.copy()
                val_parameters = data.validation.parameters.copy()
                for index, value in fixed.items():
                    train_parameters[:, index] = value
                    val_parameters[:, index] = value
                train_x = forward_model(train_parameters, generation_key)
                generation_key, validation_generation_key = jr.split(generation_key)
                val_x = forward_model(val_parameters, validation_generation_key)
            generation_seconds = time.perf_counter() - generation_start

            preprocessing_start = time.perf_counter()
            normalized_train_x, normalized_val_x, normalized_target, x_min, x_max = _global_minmax(
                train_x, val_x, target
            )
            group_train = data.train.parameters[:, list(group)]
            group_val = data.validation.parameters[:, list(group)]
            normalized_train_theta, theta_min, theta_max = _parameter_minmax(
                group_train, group
            )
            normalized_val_theta, _, _ = _parameter_minmax(group_val, group)
            preprocessing_seconds = time.perf_counter() - preprocessing_start
            model, flow_state, val_loss, epochs_trained, training_seconds = self._fit_group(
                normalized_train_x, normalized_train_theta,
                normalized_val_x, normalized_val_theta,
                group, training_key, stage_dir, normalized_target,
                theta_min, theta_max,
                data.real_test.parameters[real_test_index, list(group)], stage,
            )
            evaluation_start = time.perf_counter()
            samples = np.asarray(model.sample(
                jnp.asarray(normalized_target), sampling_key,
                self.config.posterior_samples, flow_state,
            ))[:, 0]
            samples = _parameter_inverse(samples, theta_min, theta_max)
            posteriors.append(samples.astype(np.float32))
            means = samples.mean(axis=0)
            fixed.update(zip(group, map(float, means)))
            evaluation_seconds = time.perf_counter() - evaluation_start
            np.save(stage_dir / "posterior_samples.npy", samples.astype(np.float32))
            np.savez(stage_dir / "normalization.npz", x_min=x_min, x_max=x_max,
                     theta_min=theta_min, theta_max=theta_max,
                     signal_transform="global_minmax_-1_1",
                     parameter_transform="prior_minmax_-1_1")
            (stage_dir / "metadata.json").write_text(json.dumps({
                "group": list(group), "fixed_before_stage": sorted(set(fixed) - set(group)),
                "validation_loss": val_loss,
            }, indent=2))
            timing = {
                "stage": stage + 1,
                "group": list(group),
                "epochs_trained": epochs_trained,
                "generation_seconds": generation_seconds,
                "preprocessing_seconds": preprocessing_seconds,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": time.perf_counter() - stage_start,
            }
            timings.append(timing)
            (stage_dir / "timing.json").write_text(json.dumps(timing, indent=2))
            print(f"stage {stage + 1} group={group} timing={timing}", flush=True)
            if self.tracker is not None and not self.tracker_loss_only:
                tracked = {
                    "workflow_step": self._tracking_step,
                    "stage_result/stage": stage + 1,
                    "stage_result/best_validation_loss": val_loss,
                    "stage_result/epochs_trained": epochs_trained,
                    "stage_result/training_seconds": training_seconds,
                    "stage_result/evaluation_seconds": evaluation_seconds,
                    "stage_result/total_seconds": timing["total_seconds"],
                }
                tracked.update({f"stage_result/parameter_{index}_estimate": float(value)
                                for index, value in zip(group, means)})
                self.tracker.log(tracked, step=self._tracking_step)

        estimate = np.full(parameter_dim, np.nan, dtype=np.float32)
        for index, value in fixed.items():
            estimate[index] = value
        result = CalibrationResult(estimate, data.real_test.parameters[real_test_index].copy(),
                                   tuple(posteriors), self.config.groups[:stage_count])
        np.savez(self.output_dir / "calibration_result.npz", estimate=result.estimate,
                 truth=result.truth, relative_error_percent=result.relative_error_percent)
        (self.output_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        timing_summary = {
            "workflow_seconds": time.perf_counter() - workflow_start,
            "total_generation_seconds": sum(row["generation_seconds"] for row in timings),
            "total_training_seconds": sum(row["training_seconds"] for row in timings),
            "total_evaluation_seconds": sum(row["evaluation_seconds"] for row in timings),
            "stages": timings,
        }
        (self.output_dir / "timing_summary.json").write_text(json.dumps(timing_summary, indent=2))
        if self.tracker is not None and not self.tracker_loss_only:
            self.tracker.summary["workflow_seconds"] = timing_summary["workflow_seconds"]
            self.tracker.summary["total_training_seconds"] = timing_summary["total_training_seconds"]
            self.tracker.summary["final_mean_relative_error_percent"] = float(
                np.nanmean(result.relative_error_percent)
            )
        with (self.output_dir / "timings.csv").open("w", newline="") as handle:
            fields = tuple(key for key in timings[0] if key != "group") + ("group",)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in timings:
                writer.writerow({**row, "group": " ".join(map(str, row["group"]))})
        return result
