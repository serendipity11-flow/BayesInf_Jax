"""Kalman state estimator and constrained MPC lane-keeping baseline."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .bicycle_dataset import BicycleConfig


@dataclass(frozen=True)
class MPCConfig:
    lateral_weight: float = 8.0
    heading_weight: float = 3.0
    steering_weight: float = 0.15
    steering_rate_weight: float = 1.0
    terminal_multiplier: float = 3.0
    process_lateral_std: float = 0.025
    process_heading_std: float = 0.015
    max_iterations: int = 60


class LateralKalmanFilter:
    """Linear error-state KF for ``[lateral error, heading error]``.

    Only lateral error is measured. Heading becomes observable through the
    lateral-error dynamics, so this is a genuine estimator rather than a noisy
    heading measurement being supplied to the controller.
    """

    def __init__(self, vehicle: BicycleConfig, config: MPCConfig = MPCConfig()):
        self.vehicle, self.config = vehicle, config
        self.A = np.array([[1.0, vehicle.dt * vehicle.velocity], [0.0, 1.0]])
        self.B = np.array([0.0, vehicle.dt * vehicle.velocity / vehicle.wheelbase])
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.diag([config.process_lateral_std**2, config.process_heading_std**2])
        self.R = np.array([[vehicle.obs_std**2]])
        self.mean = np.zeros(2)
        self.covariance = np.diag([0.5**2, 0.2**2])

    def reset(self, mean: Optional[np.ndarray] = None):
        self.mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=float).copy()
        self.covariance = np.diag([0.5**2, 0.2**2])

    def update(self, lateral_measurement: float, measurement_std: Optional[float] = None):
        r = self.vehicle.obs_std if measurement_std is None else max(measurement_std, 1e-6)
        innovation_cov = self.C @ self.covariance @ self.C.T + r**2
        gain = self.covariance @ self.C.T / innovation_cov.item()
        self.mean = self.mean + gain[:, 0] * (lateral_measurement - self.mean[0])
        # Joseph form retains symmetry/positive semidefiniteness.
        identity = np.eye(2)
        residual = identity - gain @ self.C
        self.covariance = residual @ self.covariance @ residual.T + (gain * r**2) @ gain.T
        return self.mean.copy()

    def predict(self, steering: float, curvature: float):
        disturbance = np.array([0.0, -self.vehicle.dt * self.vehicle.velocity * curvature])
        self.mean = self.A @ self.mean + self.B * np.tan(steering) + disturbance
        self.covariance = self.A @ self.covariance @ self.A.T + self.Q
        return self.mean.copy()


class ConstrainedLateralMPC:
    """Nonlinear receding-horizon MPC with hard steering bounds."""

    def __init__(self, vehicle: BicycleConfig, config: MPCConfig = MPCConfig()):
        self.vehicle, self.config = vehicle, config
        self.previous_plan = np.zeros(vehicle.horizon)
        self.previous_steering = 0.0
        self.last_success = True

    def reset(self):
        self.previous_plan[:] = 0.0
        self.previous_steering = 0.0
        self.last_success = True

    def _objective(self, controls: np.ndarray, state: np.ndarray,
                   curvature: np.ndarray) -> float:
        ey, ep = state
        cost = 0.0
        previous = self.previous_steering
        for i, (delta, kappa) in enumerate(zip(controls, curvature)):
            ey = ey + self.vehicle.dt * self.vehicle.velocity * np.sin(ep)
            ep = ep + self.vehicle.dt * (
                self.vehicle.velocity / self.vehicle.wheelbase * np.tan(delta)
                - self.vehicle.velocity * kappa
            )
            terminal = self.config.terminal_multiplier if i == len(controls) - 1 else 1.0
            cost += terminal * (self.config.lateral_weight * ey**2
                                + self.config.heading_weight * ep**2)
            cost += self.config.steering_weight * delta**2
            cost += self.config.steering_rate_weight * (delta - previous)**2
            previous = delta
        return float(cost)

    def command(self, estimated_state: np.ndarray, curvature_preview: np.ndarray) -> Tuple[float, np.ndarray]:
        curvature_preview = np.asarray(curvature_preview, dtype=float)
        if curvature_preview.shape != (self.vehicle.horizon,):
            raise ValueError(f"curvature preview must have shape ({self.vehicle.horizon},)")
        warm_start = np.r_[self.previous_plan[1:], self.previous_plan[-1]]
        result = minimize(
            self._objective, warm_start, args=(np.asarray(estimated_state), curvature_preview),
            method="L-BFGS-B",
            bounds=[(-float(self.vehicle.delta_max), float(self.vehicle.delta_max))] * self.vehicle.horizon,
            options={"maxiter": self.config.max_iterations, "ftol": 1e-9},
        )
        self.last_success = bool(result.success)
        # Even an iteration-limited result is feasible because L-BFGS-B always
        # respects the box bounds.
        self.previous_plan = np.clip(result.x, -self.vehicle.delta_max, self.vehicle.delta_max)
        self.previous_steering = float(self.previous_plan[0])
        return self.previous_steering, self.previous_plan.copy()


class LinearLateralMPC(ConstrainedLateralMPC):
    """Constrained MPC using the small-angle linearized error dynamics.

    The controller is linear while the common evaluation plant remains the
    nonlinear bicycle model.  This isolates the effect of model linearization
    in the comparison with :class:`ConstrainedLateralMPC`.
    """

    def _objective(self, controls: np.ndarray, state: np.ndarray,
                   curvature: np.ndarray) -> float:
        ey, ep = state
        cost = 0.0
        previous = self.previous_steering
        dt_v = self.vehicle.dt * self.vehicle.velocity
        steering_gain = dt_v / self.vehicle.wheelbase
        for i, (delta, kappa) in enumerate(zip(controls, curvature)):
            ey = ey + dt_v * ep
            ep = ep + steering_gain * delta - dt_v * kappa
            terminal = self.config.terminal_multiplier if i == len(controls) - 1 else 1.0
            cost += terminal * (
                self.config.lateral_weight * ey**2
                + self.config.heading_weight * ep**2
            )
            cost += self.config.steering_weight * delta**2
            cost += self.config.steering_rate_weight * (delta - previous)**2
            previous = delta
        return float(cost)


# Descriptive alias retained alongside the original public name.
NonlinearLateralMPC = ConstrainedLateralMPC
