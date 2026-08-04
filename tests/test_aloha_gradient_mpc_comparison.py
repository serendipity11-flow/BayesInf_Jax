import jax.numpy as jnp
from experiments.aloha_optimal_control.gradient_mpc_comparison import (
    SecondOrderServo,
)
from bayesinf.applications.optimal_control.gradient_based_mpc import (
    GradientMPCConfig,
)


def test_servo_converts_action_normalization_to_state_normalization():
    model = SecondOrderServo(
        momentum=jnp.array([0.5]), command_gain=jnp.array([0.1]),
        action_to_state_scale=jnp.array([2.0]),
        action_to_state_bias=jnp.array([3.0]), history_len=2,
    )
    history = jnp.array([[-0.2], [0.0]])
    # normalized action 0.5 represents normalized state command 2*0.5+3=4
    predicted = model.predict_next(history, jnp.array([0.5]))
    assert jnp.allclose(predicted, jnp.array([0.5]))  # .5*.2 + .1*4


def test_gradient_config_rejects_invalid_values():
    try:
        GradientMPCConfig(horizon=0)
        raise AssertionError("zero horizon was accepted")
    except ValueError:
        pass
    try:
        GradientMPCConfig(momentum=1.0)
        raise AssertionError("unit momentum was accepted")
    except ValueError:
        pass
