# Differential-drive FlowSDE

This example learns the controlled stochastic dynamics of a two-wheel MuJoCo robot.
The observation is `[p_x, p_y, theta, v, omega]`, and `[v_cmd, omega_cmd]` is fed to
both the drift MLP and the conditional RealNVP diffusion network. The navigation arena
has a smooth two-dimensional traction field (`mu` is approximately 0.48--0.88), avoiding
an unobservable, discontinuous ice boundary while retaining spatially varying slip and
nonlinear wheel-ground dynamics. Friction is never a model input; the HDF5 value is for
stratified training diagnostics only.

Generate data and train:

```bash
python -m experiments.differential_drive_flowsde.dataset \
  --output data/differential_drive/smooth_terrain_navigation.h5
python -m experiments.differential_drive_flowsde.train_multishoot \
  --data data/differential_drive/smooth_terrain_navigation.h5 \
  --output results/results_optimal_control/differential_drive/smooth_terrain_seed0 \
  --terrain-bins 4
python -m experiments.differential_drive_flowsde.evaluate \
  --data data/differential_drive/smooth_terrain_navigation.h5 \
  --result results/results_optimal_control/differential_drive/smooth_terrain_seed0/joint_stage \
  --output results/results_optimal_control/differential_drive/smooth_terrain_seed0/evaluation
python -m experiments.differential_drive_flowsde.evaluate_closed_loop \
  --result results/results_optimal_control/differential_drive/smooth_terrain_seed0/joint_stage \
  --output results/results_optimal_control/differential_drive/smooth_terrain_seed0/evaluation
```

For a quick end-to-end smoke run:

```bash
python -m experiments.differential_drive_flowsde.dataset --output /tmp/dd.h5 \
  --trajectories 10 --split-trajectories 7 2 1 --steps 50
python -m experiments.differential_drive_flowsde.train --data /tmp/dd.h5 \
  --output /tmp/dd-result --drift-epochs 1 --joint-epochs 1 \
  --samples-per-epoch 128 --validation-samples 32 --width 16 --condition-dim 8 \
  --flow-layers 2 --wandb-project ""
```

Stage 1 optimizes only deterministic one-step drift MSE. Since the flow is absent from
that computation, its parameters receive no gradients. Stage 2 continues from that
checkpoint and jointly optimizes transition likelihood plus a drift-retention penalty.
Angle increments are wrapped before normalization to avoid discontinuities at `+-pi`.

`evaluate.py` is the open-loop benchmark: one-step likelihood/calibration plus held-out
recorded-command rollouts. `evaluate_closed_loop.py` uses the learned drift inside a
receding-horizon CEM-MPC controller on three diagonal waypoint tasks. It reports success
rate and final goal error and plots trajectories over the traction field. The SLURM
script `slurm_train_differential_drive_run_b.sh` runs all four stages from fresh data.

## Matched nonlinear-physics and FlowSDE closed-loop comparison

The state-matched physical baseline uses
`[p_x, p_y, theta, v_forward, v_lateral, omega, wheel_left, wheel_right]`.  Its dynamic
update is a ridge-identified nonlinear model of the five velocity/wheel states.  The
feature library contains linear actuator and state terms, quadratic drag, body coupling,
wheel sum/difference, position-dependent traction, and traction-command interactions.
Pose is not independently regressed: midpoint body velocities and yaw rate are integrated
through the exact planar rigid-body kinematics.  State increments are clipped to the
training range plus a ten-percent margin.  Gaussian process noise is estimated as the
full covariance of the training residuals and added only to the five modeled dynamic
increments before pose integration.

The matched MPPI settings use 64 sampled input sequences and 64 stochastic rollouts for
each input sequence.  FlowSDE uncertainty is generated solely by its learned conditional
flow: standard-normal latent samples are transformed by the flow and are not an extra
Gaussian state perturbation.  The physical baseline instead transforms standard-normal
samples with its fitted residual Cholesky factor.  Both controllers minimize the same
expected cost: mean waypoint distance plus 2.5 times terminal distance, 0.015 times mean
weighted action effort, and 0.02 times mean action roughness.
