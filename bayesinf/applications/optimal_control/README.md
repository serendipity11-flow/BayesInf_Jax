# Optimal-control methods

This package contains controlled probabilistic dynamics, sampling MPC,
projected-gradient MPC, ALOHA experiments, and lane-keeping experiments.
The ALOHA implementation and its validation boundary are documented first;
lane-keeping documentation follows afterward.

## ALOHA Mobile Cabinet forward dynamics

### Controlled RealNVP FlowSDE

`train_aloha_flow_sde.py` implements the causal controlled SDE

```text
dx[t] = f(x[t], u[t], t) dt + g(epsilon; x[t], u[t], t) sqrt(dt)
```

Pass `--diffusion-no-control` to instead use
`g(epsilon; x[t], t)`, forcing all command-dependent mean dynamics through
`f(x[t], u[t], t)`. Checkpoints for this ablation include `_gx` in the name.

with `dt=0.02` s. The RealNVP diffusion is centred so that `g(0)=0`; this
prevents its affine translations from absorbing the deterministic drift.
Normalized input states and controls are perturbed during training (both use
standard deviation `0.01` by default), while next-state labels remain clean.
The train/validation/test split is by complete episode.

```bash
python -m bayesinf.applications.optimal_control.train_aloha_flow_sde \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --epochs 20 --batch-size 8 --window-steps 128 \
  --segment-length 8 --solver heun --train-particles 4

python -m bayesinf.applications.optimal_control.eval_aloha_flow_sde \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --config results/results_optimal_control/aloha_mobile_cabinet/flow_sde/aloha_flow_sde_heun_seg8_noise0p01_seed0_config.json \
  --particles 64
```

The evaluator recursively predicts all 1,499 transitions of every held-out
episode and writes aggregate/per-joint accuracy, interval coverage, predictive
spread, horizon RMSE, and trajectory/calibration figures.

### FlowSDE MPPI versus physical-model MPPI

`compare_aloha_flow_sde_physical_mppi.py` evaluates both controllers with the
same MPPI perturbations, bounds, cost, horizon, and fitted-servo execution
plant. Learned-model candidates are stochastic FlowSDE paths integrated by the
solver stored in the training configuration; physical-model candidates use
deterministic fitted-servo recursion.

```bash
python -m bayesinf.applications.optimal_control.compare_aloha_flow_sde_physical_mppi \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --flow-config results/results_optimal_control/aloha_mobile_cabinet/flow_sde_dense_uw0p8/aloha_flow_sde_heun_seg16_noise0p01_seed0_config.json \
  --solver heun --candidates 128 --particles 4 --iterations 3
```

This remains an offline diagnostic: the common execution plant is a fitted
servo, not ALOHA hardware or a high-fidelity simulator. The physical-model
controller therefore has an inherent model-matching advantage.

The ALOHA runner uses all 14 state and action channels. It splits by episode,
trains on fixed segments, and evaluates by recursively rolling from the first
measured history to the end of every complete test episode:

```bash
python -m bayesinf.applications.optimal_control.train_aloha_mobile_dynamics \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --flow-type realnvp --state-noise-std 0.05 \
  --history 8 --segment-steps 16 --stride 16 --epochs 20
```

Its flow condition is exactly `[GRU(x_history)[-1], u_t]`; only the final GRU
hidden state is used, and previous actions are not inputs to the encoder.
State noise is applied to normalized encoder contexts during training while
the next-state supervision remains clean.
Checkpoints, normalization, learning curves, aggregate metrics, and per-step
full-horizon RMSE are saved under
`results/results_optimal_control/aloha_mobile_cabinet/`.

### Dataset and alignment

The LeRobot parquet contains 85 episodes of 1,500 samples at 50 Hz. Both
`observation.state` and `action` have 14 channels in left-arm then right-arm
motor order, including the two grippers. Row `t` is interpreted as state
`x[t]` and the command applied for transition `x[t] -> x[t+1]`; the last action
in each episode is dropped because it has no recorded next-state target.

Episodes, rather than windows, are shuffled into 68 training, 8 validation and
9 test episodes. This prevents overlapping temporal windows from leaking
between splits. State and action normalization statistics are computed only
from training episodes and are stored beside each checkpoint.

For the reported run, each training item has 24 state samples and 23 actions:

```text
8 history states + 16 supervised transitions = 24 states
```

Segments use a stride of 16. Within a segment, transition `k` conditions on
the moving eight-state window ending at the current state and on only the
current action.

### Dynamics distribution

`GRULatentSSMFlow` models

```text
p(x[t+1] | GRU(x[t-7:t])[-1], u[t]).
```

Only the final GRU hidden state is concatenated with `u[t]`. The invertible
decoder predicts normalized state increments. RealNVP is the default; NSF is
available through `--flow-type nsf`. Latents use a stationary linear Gaussian
SSM, so sampled rollout disturbances are temporally correlated rather than
independently redrawn transition densities.

With `--state-noise-std 0.05`, Gaussian noise is applied to normalized state
contexts seen by the encoder. Next-state labels stay clean, and the residual is
formed from the noisy current state to the clean next state. This is denoising
input augmentation, not label-noise augmentation.

### Jackal obstacle avoidance

The Jackal implementation uses the causal controlled transition

```text
p(x[t+1] | x[t-H+1:t], u[t])
x = [world_x, world_y, sin(yaw), cos(yaw), v, omega]
u = [v_command, omega_command]
```

Train first with the included simulator smoke test:

```bash
python -m bayesinf.applications.optimal_control.train_jackal_dynamics \
  --epochs 30 --history 5 --rollout-steps 5
```

For robot data, export an NPZ with `states` shaped `(episodes,T+1,6)` and
`actions` shaped `(episodes,T,2)`, then pass `--data jackal.npz`. An action at
index `t` must be the command active between state timestamps `t` and `t+1`;
interpolate odometry to the command clock before exporting.

At control time, construct `obstacle_cost(goal, obstacles)` where each obstacle
is `(x, y, radius)`, then use `CEMStochasticMPC` or `MPPIController` with
`action_low=[0,-1.5]`, `action_high=[1,1.5]`. Keep the latest five state samples
and four prior commands, call `controller.command(history, prior_commands,key)`,
publish only the returned first command, and repeat at 10 Hz. The MPC
`risk_weight` should be positive (start at `1.0`) so candidates with uncertain
collision outcomes are rejected.

### Evaluation protocol

Segment NLL is measured on held-out segments. Open-loop evaluation supplies
the first eight true states once, then recursively predicts the remaining
1,492 states (29.84 s). The evaluator reports zero-latent nominal error,
particle-mean error, per-joint error, horizon-wise RMSE, empirical interval
coverage and interval width. Since long-horizon joint errors mix channels with
different physical ranges, per-joint metrics should be consulted alongside the
aggregate RMSE.

```bash
python -m bayesinf.applications.optimal_control.eval_aloha_mobile_dynamics \
  --data data/aloha_mobile_cabinet/file-000.parquet --particles 64
```

### Short-horizon MPC comparison

`aloha_gradient_mpc_comparison.py` compares a nominal learned-flow controller
with a deliberately simple second-order servo controller. Both use projected
gradient descent, the same 10-step (0.2 s) horizon, box constraints and cost.
Because the recorded dataset contains no counterfactual outcome for a newly
optimized command, both controllers are executed against the fitted servo
plant. This is an offline optimizer/model diagnostic, not an ALOHA hardware or
high-fidelity simulator result.

For normalized state `x_n` and normalized action `u_n`, the servo first maps
the independently normalized action into state coordinates:

```text
u_as_state = (u_std / x_std) * u_n + (u_mean - x_mean) / x_std
x_next = x + a * (x - x_previous) + b * (u_as_state - x)
```

The affine conversion is essential: directly subtracting independently
normalized state and action values is incorrect. Coefficients `a` and `b` are
fit per joint on training episodes and constrained to stable ranges. The
comparison reports tracking and terminal RMSE, normalized control effort,
one-step action-Jacobian norm and median solve time.

```bash
python -m bayesinf.applications.optimal_control.aloha_gradient_mpc_comparison \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --horizon 10 --iterations 25 --tasks 3
```

`aloha_mpc_optimizer_comparison.py` crosses both dynamics models with both
optimizers: flow+gradient, flow+MPPI, physical+gradient and physical+MPPI. MPPI
uses three sampling updates per command, 128 action-sequence candidates and
four temporally correlated flow particles per candidate. Candidate costs use
mean plus `risk_weight * standard_deviation`. Because stochastic-flow and
deterministic-physical costs have different raw scales, the comparison divides
cost differences by their candidate-set standard deviation before applying the
dimensionless softmax temperature. Effective sample size is reported to expose
weight collapse.

```bash
python -m bayesinf.applications.optimal_control.aloha_mpc_optimizer_comparison \
  --data data/aloha_mobile_cabinet/file-000.parquet \
  --horizon 10 --candidates 128 --particles 4 \
  --mppi-iterations 3 --gradient-iterations 25 --tasks 3
```

In the seed-0 three-task offline comparison, physical+MPPI performs best
(`0.192` tracking RMSE), followed by physical+gradient (`0.289`),
flow+gradient (`0.522`) and flow+MPPI (`0.566`). Flow+MPPI improves its effective
sample size from `1.33/128` without scaling to `11.08/128` with cost scaling,
but it still moves the shared plant in less useful directions. Thus sampling
does bypass the flow's weak derivative, but cannot repair incorrect or weakly
identified counterfactual action effects in the learned dynamics.

### High-fidelity ALOHA simulation status

The generic `BraxGradientMPC` is implemented and tested on a Brax environment,
but a high-fidelity ALOHA plant is **not yet integrated in this repository**.
The intended future comparison is:

```text
high-fidelity ALOHA MuJoCo/MJX plant (hidden from controllers)
  <- action from learned-flow MPC or simplified-physics MPC
  -> shared simulated joint/contact observation
```

That experiment requires the official ALOHA MJCF assets, an explicit mapping
between the dataset's 14 channels and simulator actuators, cabinet/task
geometry, matching control rate and actuator semantics, and validation of the
initial-state reset. Until those pieces exist, results from the fitted servo
must not be described as simulated ALOHA performance.

### File map

- `gru_latent_ssm_flow.py`: GRU context, exact flow likelihood and rollouts.
- `train_aloha_mobile_dynamics.py`: episode split, segments, noise and training.
- `eval_aloha_mobile_dynamics.py`: full-horizon accuracy and uncertainty.
- `gradient_based_mpc.py`: learned/analytic and Brax gradient MPC.
- `sampling_mpc.py`: CEM and MPPI stochastic MPC only.
- `aloha_gradient_mpc_comparison.py`: fitted-servo offline comparison.
- `aloha_mpc_optimizer_comparison.py`: four-way optimizer/model comparison.

### Verification performed

The implementation is covered by executable checks for episode-disjoint
splits, state/action segment alignment, RealNVP and NSF likelihoods, clean-label
noisy-context training, deterministic and stochastic rollout shapes, initial
state anchoring, final-GRU-state conditioning, CEM/MPPI/gradient action bounds,
invalid gradient-MPC configuration, action-to-state normalization in the servo,
finite action gradients, checkpoint reconstruction, and differentiation through
a real Brax environment rollout. The corrected three-task servo comparison is
stored in `results/results_optimal_control/aloha_mobile_cabinet/mpc_comparison/`.

Known warnings are limited to Equinox reporting NumPy/JAX coupling-index arrays
as static and Brax reporting that its legacy environment physics is no longer
actively maintained. Neither warning changed the finite-value or gradient
checks. New high-fidelity ALOHA work should use the official MJX-compatible
ALOHA model rather than treating the generic Brax test environment as ALOHA.

## Gradient MPC and Brax

Sampling controllers are implemented in `sampling_mpc.py`; projected-gradient
controllers are separate in `gradient_based_mpc.py`. `GradientMPC` rolls
through a learned or analytic array-based model, while `BraxGradientMPC`
differentiates directly through a Brax environment state and `env.step`:

```python
from brax import envs
import jax.numpy as jnp
import jax.random as jr
from bayesinf.applications.optimal_control import (
    BraxGradientMPC, GradientMPCConfig,
)

env = envs.get_environment("inverted_pendulum", backend="generalized")
state = env.reset(jr.PRNGKey(0))
cost = lambda next_state, action, t: (
    jnp.sum(next_state.obs**2) + 1e-3 * jnp.sum(action**2)
)
mpc = BraxGradientMPC(
    env, cost, GradientMPCConfig(horizon=10, iterations=25)
)
action, plan = mpc.command(state)
```

The repository has no ALOHA MJCF/URDF. Consequently the ALOHA comparison uses
the fitted joint-servo plant; using Brax for ALOHA requires supplying and
validating a matching robot model and actuator configuration.

This experiment replaces an explicit state estimator and MPC solve with one
conditional distribution over a steering chunk. A causal GRU encodes aligned
past noisy lateral errors and applied controls (implicit estimation); a separate
GRU encodes future lane
displacement, heading and curvature (preview planning). A conditional RealNVP
maps both embeddings to `H` steering commands. A `tanh` output transform enforces
bounded steering, followed by clipping to the physical `35 deg` limit.

Training paths are differentiably-flat sinusoidal centerlines. Broad lateral
and heading recovery states are paired with dynamically consistent histories.
Target chunks are produced by a bounded nonlinear MPC optimizer using the same
nonlinear bicycle dynamics as the evaluation plant.

Run from the repository root:

```bash
python -m bayesinf.applications.optimal_control.train_lane_keeping --limit
python -m bayesinf.applications.optimal_control.evaluate_lane_keeping \
  --checkpoint results/optimal_control/lane_flow.eqx
```

Remove `--limit` for the default 12,000-window experiment. Evaluation uses an
unseen winding path, executes the first command from each predicted chunk in a
receding-horizon loop, sweeps sensor noise, and reports lateral RMSE, steering
RMS, maximum steering, and constraint violations. It compares against a
warm-started nonlinear finite-horizon MPC with hard steering bounds. The MPC is
fed by a Kalman filter whose only measurement is noisy lateral deviation; its
hidden heading-error estimate is recovered through the process model. Both
controllers receive the identical path and noise samples. Results, including
steering-rate RMS, optimizer failures, and per-command runtime, are written to
`results/optimal_control/comparison.csv`.

The flow is probabilistic. `sample(..., deterministic=True)` uses the latent
median for control; stochastic samples expose action uncertainty and can be
used for risk-sensitive selection in later experiments.

Saturation-aware flow
---------------------

The nonlinear-MPC expert frequently places recovery controls exactly at the
physical steering bounds, `delta = +/- delta_max`. With the original transform

```text
delta = delta_max * tanh(v),
```

an exact boundary target is only reached as the unconstrained flow output `v`
approaches infinity. This makes concentrated saturated targets harder to
represent and weakens gradients near the limit.

The saturation-aware variant enlarges only the flow's internal statistical
support:

```text
internal_bound = saturation_factor * delta_max
delta_internal = internal_bound * tanh(v)
delta_applied  = clip(delta_internal, -delta_max, delta_max)
```

The experiment uses `saturation_factor=1.05`. The physical bound then occurs at
a finite unconstrained value:

```text
atanh(delta_max / internal_bound) = atanh(1 / 1.05) = 1.857.
```

The likelihood uses `internal_bound` for its inverse transform and Jacobian,
while history controls remain normalized by the physical `delta_max`. Sampling
and deterministic execution are always clipped to the physical limit, so the
method does not introduce steering-constraint violations.

Train and evaluate this variant with:

```bash
python -m bayesinf.applications.optimal_control.train_lane_keeping \
  --first-action-weight 0 \
  --rollout-weight 0 \
  --saturation-factor 1.05 \
  --output results/results_optimal_control/runs/seed0_saturation_factor_105/lane_flow.eqx

python -m bayesinf.applications.optimal_control.evaluate_lane_keeping \
  --checkpoint results/results_optimal_control/runs/seed0_saturation_factor_105/lane_flow.eqx \
  --saturation-factor 1.05
```

The corresponding Slurm launcher is
`slurm_train_saturation_aware_flow.sh`.

Observed seed-0 trade-off
-------------------------

Under the corrected closed-loop evaluator, the `1.05` variant improves the
standard noise-sweep mean RMSE from `0.0991 m` to `0.0977 m`. It also reduces
the worst RMSE over the 15-state initial-condition sweep from `0.1962 m` to
`0.1857 m`, including improvement in the strongly saturated recovery corners.
Its average 15-state RMSE is slightly worse (`0.1107 m` versus `0.1086 m`),
because several interior-control cases lose a few millimeters of accuracy.

Thus the saturation-aware method is useful when nominal-noise or worst-case
recovery performance is prioritized. The original recovery/NMPC flow remains
slightly better on average across the complete initial-condition grid.

Additional entry points
-----------------------

- `evaluate_direct_control_prediction.py`: held-out control comparison without rollout.
- `evaluate_initial_condition_sweep.py`: 15-case closed-loop robustness sweep.
- `plot_closed_loop_controller_comparison.py`: one-run flow/linear-MPC/nonlinear-MPC plot.
- `plot_best_closed_loop_comparison.py`: published best-run noise-sweep comparison.
- `plot_multiseed_control_results.py`: aggregate metrics across seed directories.
- `slurm_train_recovery_nmpc_flow.sh`: best recovery/NMPC training configuration.
- `slurm_train_control_aware_flow.sh`: auxiliary first-action/rollout-loss experiment.
- `slurm_train_saturation_aware_flow.sh`: enlarged internal-bound saturation experiment.
- `slurm_train_multiseed_lane_flow.sh`: five-seed Slurm array.

Run Python tools as modules, for example:

```bash
python -m bayesinf.applications.optimal_control.evaluate_direct_control_prediction
python -m bayesinf.applications.optimal_control.plot_closed_loop_controller_comparison
```
