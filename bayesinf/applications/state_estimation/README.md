# State-estimation framework

This package contains reusable flow estimators, stochastic solvers, and
classical filters. Dataset-specific training and evaluation entry points live
in direct use-case packages under `experiments`.

- `amortized_state_flow.py`, `pointwise_state_flow.py`, `latent_ssm_flow.py`,
  `flow_sde.py`, and `stochastic_interpolant.py`: flow-based estimators.
- `filtering_baselines.py`: model-agnostic EKF, UKF, and bootstrap particle
  filtering. Callers provide their transition function.
- `stochastic_solvers.py`: reusable stochastic integration.
- `utils/plots.py`: generic estimation diagnostics shared by experiments.

ALOHA insertion and Lorenz-96 datasets, trainers, evaluators, and plotting
utilities are under `experiments/aloha_state_estimation` and
`experiments/lorenz96_state_estimation`.
