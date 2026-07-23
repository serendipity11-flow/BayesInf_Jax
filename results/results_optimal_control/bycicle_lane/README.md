# Optimal-control best result

This directory contains the seed-0 recovery/NMPC flow, the saturation-aware
variant, and their corrected comparisons with linear and nonlinear MPC. Both
flows use aligned past controls, dynamically consistent recovery histories,
and nonlinear-MPC target chunks. The control-aware-loss run is not included
because it produced slightly worse closed-loop tracking.

Key files:

- `amortized_seq_flow_recovery_nmpc_checkpoint.eqx`: recovery/NMPC Equinox checkpoint.
- `runs/seed0_saturation_factor_105/lane_flow.eqx`: saturation-aware checkpoint.
- `amortized_seq_flow_recovery_nmpc_config.txt`: vehicle and training-data configuration.
- `amortized_seq_flow_recovery_nmpc_closed_loop_metrics.csv`: original published noise-sweep metrics.
- `amortized_seq_flow_recovery_nmpc_direct_control_metrics.csv`: held-out control-prediction metrics without rollout.
- `amortized_seq_flow_recovery_nmpc_direct_prediction.png`: direct expert-versus-flow prediction diagnostics.
- `amortized_seq_flow_before_after_recovery_nmpc.png`: recovery/NMPC flow compared with MPC across noise levels.
- `amortized_seq_flow_cs_linear_nonlinear_mpc_initial_condition_sweep.csv`: original-flow 15-state metrics.
- `amortized_seq_flow_cs_linear_nonlinear_mpc_initial_condition_sweep.png`: original-flow RMSE heatmaps.
- `amortized_seq_flow_cs_linear_nonlinear_mpc_initial_condition_performance.png`: original-flow accuracy and runtime summary.
- `amortized_seq_flow_cs_linear_nonlinear_mpc_initial_condition_trajectories.png`: original-flow representative trajectories.
- `all_flow_variants_noise_sweep.csv` / `.png`: corrected comparison of both available flow variants and both MPC baselines across noise levels.
- `all_flow_variants_initial_condition_sweep.csv`: unified 15-state metrics for both flow variants and MPC baselines.
- `all_flow_variants_initial_condition_performance.png`: variant RMSE distributions and per-state differences from nonlinear MPC.
- `all_flow_variants_initial_condition_trajectories.png`: representative trajectories for both flows and both MPC controllers.

Saturation-aware method
-----------------------

The saturation-aware flow uses an internal bound of `1.05 * delta_max` and
clips executed controls to the physical `delta_max`. This places an exact
physical-bound target at the finite unconstrained value `atanh(1/1.05)=1.857`,
instead of requiring the `tanh` input to approach infinity. It preserves zero
constraint violations while better representing saturated recovery actions.

Across the corrected three-noise sweep, its mean RMSE is `0.0977 m`, compared
with `0.0991 m` for the original flow. Across 15 initial states, it improves
worst-case RMSE from `0.1962 m` to `0.1857 m`, but slightly worsens mean RMSE
from `0.1086 m` to `0.1107 m`. The unified `all_flow_variants_*` files are the
recommended results for comparing this trade-off.
