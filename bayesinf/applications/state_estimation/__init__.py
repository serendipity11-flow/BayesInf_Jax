"""
State estimation with normalizing flows.

- AmortizedStateFlow: Amortized Bayesian state inference; encoder + flow over
  state sequences with Wiener-process latent prior.
- FlowSDE: exact conditional RealNVP or scaled-NICE transition NLL plus
  stochastic sampling and a bidirectional sequence encoder.
- LatentSSMFlow: Linear Gaussian SSM in latent z-space with a conditional
  normalizing flow decoder; exact MLE training via change of variables.
- NeuralStochasticFlow: solver-free finite-time conditional transition flow.
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
from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    ConsistencyLosses,
    NSF,
    NSFBridge,
    NSFConfig,
    NSFConsistencyConfig,
    NSFConsistencyTrainingResult,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    evaluate_ou_moments,
    fit_normalization,
    flow_consistency_loss,
    make_step_nsf,
    make_step_nsf_bridge,
    make_step_nsf_consistency,
    make_transition_pairs,
    sample_uniform_consistency_triplets,
    sample_ou_transition_data,
    train_nsf_nll,
    train_nsf_with_consistency,
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
    "ConsistencyLosses",
    "NSF",
    "NSFBridge",
    "NSFConfig",
    "NSFConsistencyConfig",
    "NSFConsistencyTrainingResult",
    "NSFTrainingConfig",
    "NeuralStochasticFlow",
    "OrnsteinUhlenbeck",
    "evaluate_ou_moments",
    "fit_normalization",
    "flow_consistency_loss",
    "make_step_nsf",
    "make_step_nsf_bridge",
    "make_step_nsf_consistency",
    "make_transition_pairs",
    "sample_uniform_consistency_triplets",
    "sample_ou_transition_data",
    "train_nsf_nll",
    "train_nsf_with_consistency",
]
from bayesinf.applications.state_estimation.filtering_baselines import (
    FilterResult,
    ParticleFilterConfig,
    bootstrap_nsf_particle_filter,
    bootstrap_particle_filter,
    bootstrap_particle_filter_nsf,
    extended_kalman_filter,
    reconstruct_particle_trajectories,
    unscented_kalman_filter,
)
from bayesinf.applications.state_estimation.observation_models import (
    IdentityGaussianObservation,
    IdentityGaussianObservationModel,
    LinearGaussianObservation,
    LinearGaussianObservationModel,
    NonlinearGaussianObservation,
    NonlinearGaussianObservationModel,
    ObservationModel,
)
from bayesinf.applications.state_estimation.transition_benchmarks import (
    BenchmarkTrainingConfig,
    NSFTransitionAdapter,
    NormFlowSDETransitionAdapter,
    TransitionKernel,
    count_trainable_parameters,
    ensemble_transition_metrics,
    train_transition_kernel_nll,
)
from bayesinf.applications.state_estimation.controlled_systems import (
    ControlledDuffingOscillator,
    ControlledStochasticPendulum,
    sample_controlled_duffing_data,
    sample_controlled_pendulum_data,
)

__all__ += [
    "FilterResult",
    "ParticleFilterConfig",
    "bootstrap_nsf_particle_filter",
    "bootstrap_particle_filter",
    "bootstrap_particle_filter_nsf",
    "extended_kalman_filter",
    "reconstruct_particle_trajectories",
    "unscented_kalman_filter",
    "IdentityGaussianObservation",
    "IdentityGaussianObservationModel",
    "LinearGaussianObservation",
    "LinearGaussianObservationModel",
    "NonlinearGaussianObservation",
    "NonlinearGaussianObservationModel",
    "ObservationModel",
    "NSFTransitionAdapter",
    "NormFlowSDETransitionAdapter",
    "TransitionKernel",
    "count_trainable_parameters",
    "ensemble_transition_metrics",
    "BenchmarkTrainingConfig",
    "train_transition_kernel_nll",
    "ControlledDuffingOscillator",
    "ControlledStochasticPendulum",
    "sample_controlled_duffing_data",
    "sample_controlled_pendulum_data",
]
