# MuJoCo narrow-passage pushing for FlowSDE MPC/MPPI

This use case contains a three-joint planar robot arm that pushes a cuboid
through a narrow passage toward a target position and yaw. The generated
transition data includes correlated pseudo-process force and torque applied to
the object, smooth exploratory controls, and decomposed task costs.

Generate the default dataset:

```bash
python3 experiments/mujoco_pusher_flowsde/generate_data.py
```

The parquet rows contain `observation.state`, `action`, `next_state`,
`object_pose`, `object_contact_force`, `pseudo_process_noise`, the disturbance
regime, and individual goal, obstacle, control, and smoothness costs.
`object_contact_force` is MuJoCo's net generalized contact wrench on the
object's planar `(x, y, yaw)` DoFs: `[Fx, Fy, torque_yaw]`. It corresponds to
the transition from `observation.state` to `next_state`. The adjacent JSON file
records dimensions, units, timing, and generation parameters.

Train and evaluate the causal controlled FlowSDE:

```bash
sbatch scripts/slurm_train_mujoco_pusher_flowsde.sh
```

The training uses 100-step trajectory windows, 5-step stochastic
multiple-shooting segments, and conditions both deterministic drift and
RealNVP diffusion on the robot control. The pusher job also uses the contact
wrench as privileged, training-only weighting of both the `q4:q6`/`v4:v6`
one-step loss and a sampled multi-step energy score over the complete object
trajectory. The latter supervises the accuracy and spread of stochastic object
rollouts through contact and its aftermath, rather than only teacher-forced
transitions. Contact force and future object states are targets only: neither is
supplied as a model input or needed during evaluation.
