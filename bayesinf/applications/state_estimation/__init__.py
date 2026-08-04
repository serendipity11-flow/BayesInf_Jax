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
]
from bayesinf.applications.state_estimation.filtering_baselines import (
    FilterResult,
    bootstrap_particle_filter,
    extended_kalman_filter,
    unscented_kalman_filter,
)

__all__ += [
    "FilterResult",
    "bootstrap_particle_filter",
    "extended_kalman_filter",
    "unscented_kalman_filter",
]
