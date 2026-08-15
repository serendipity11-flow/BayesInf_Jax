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
