"""
State estimation with normalizing flows.

- AmortizedStateFlow: Amortized Bayesian state inference; encoder + flow over
  state sequences with Wiener-process latent prior.
- FlowSDE: exact conditional RealNVP or scaled-NICE transition NLL plus
  stochastic sampling and a bidirectional sequence encoder.
- LatentSSMFlow: Linear Gaussian SSM in latent z-space with a conditional
  normalizing flow decoder; exact MLE training via change of variables.
"""

from bayesinf.applications.state_estimation.amortized_state_flow import (
    GRUSequenceEncoder,
    TransformerSequenceEncoder,
    StateConditionEncoder,
    AmortizedStateFlow,
    fm_loss,
    sbm_loss,
    state_flow_loss,
)
from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPFlowSDE,
    NICEDiffusionNet,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    VirtualBrownianTree,
    get_solver,
    list_solvers,
    integrate_sde,
    integrate_sde_multiple_particles,
)
from bayesinf.applications.state_estimation.stochastic_interpolant import (
    StochasticInterpolantSE,
    make_step_si,
    make_step_si_ms,
)
from bayesinf.applications.state_estimation.latent_ssm_flow import (
    LatentSSMFlow,
    build_latent_ssm_flow,
    _flow_fwd_single,
    _flow_inv_single,
)
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    lorenz96_rhs,
    integrate_lorenz96,
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
    state_mae,
)

__all__ = [
    "RealNVPFlowSDE",
    "NICEDiffusionNet",
    "VirtualBrownianTree",
    "get_solver",
    "list_solvers",
    "integrate_sde",
    "integrate_sde_multiple_particles",
    "StochasticInterpolantSE",
    "make_step_si",
    "make_step_si_ms",
    "GRUSequenceEncoder",
    "TransformerSequenceEncoder",
    "StateConditionEncoder",
    "AmortizedStateFlow",
    "fm_loss",
    "sbm_loss",
    "state_flow_loss",
    "LatentSSMFlow",
    "build_latent_ssm_flow",
    "_flow_fwd_single",
    "_flow_inv_single",
    "lorenz96_rhs",
    "integrate_lorenz96",
    "generate_lorenz96_dataset",
    "train_val_test_split",
    "state_rmse",
    "state_mae",
]
