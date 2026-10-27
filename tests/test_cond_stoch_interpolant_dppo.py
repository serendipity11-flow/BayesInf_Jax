import pytest

torch = pytest.importorskip("torch", reason="PyTorch SI baselines are optional")

from experiments.cond_stoch_interpolant_dppo.policies import (
    CondStochInterpolant,
    parameter_report,
)


def test_interpolant_shapes_gradients_and_finite_logprobs():
    torch.manual_seed(0)
    model = CondStochInterpolant(39, 28, horizon=4, steps=5)
    cond = {"state": torch.randn(3, 1, 39)}
    actions = torch.randn(3, 4, 28).clamp(-1, 1)
    loss = model.loss(actions, cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    sample = model.sample_chain(cond)
    assert sample.trajectories.shape == (3, 4, 28)
    assert sample.chains.shape == (3, 6, 4, 28)
    logp = model.transition_log_prob(cond, sample.chains)
    assert logp.shape == (3, 5, 4, 28)
    assert torch.isfinite(logp).all()


def test_gaussian_base_ablation_has_same_velocity_capacity():
    informative = CondStochInterpolant(39, 28, 4, 5, informative_base=True)
    gaussian = CondStochInterpolant(39, 28, 4, 5, informative_base=False)
    assert parameter_report(informative.velocity) == parameter_report(gaussian.velocity)
    assert informative.base is not None and gaussian.base is None

