import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.filtering_baselines import (
    bootstrap_particle_filter,
    extended_kalman_filter,
    unscented_kalman_filter,
)
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    rk4_step,
)


TRANSITION = lambda x: rk4_step(x, 0.01, 8.0)


def _data(noise="gaussian"):
    return generate_lorenz96_dataset(
        jr.PRNGKey(0), 2, 5, N=4, obs_interval=2, obs_std=.1,
        obs_df=5.0 if noise == "student_t" else None,
        obs_noise_type=noise, dt=.01, spinup_steps=2,
    )


def test_ekf_shapes_and_finite_covariance():
    d = _data()
    r = extended_kalman_filter(d["y"], d["x0"], d["H"], TRANSITION, obs_std=.1)
    assert r.mean.shape == (2, 5, 4)
    assert r.covariance.shape == (2, 5, 4, 4)
    assert bool(jnp.all(jnp.isfinite(r.mean)))
    assert bool(jnp.all(jnp.diagonal(r.covariance, axis1=-2, axis2=-1) >= 0))


def test_ukf_shapes_and_finite_covariance():
    d = _data()
    r = unscented_kalman_filter(d["y"], d["x0"], d["H"], TRANSITION, obs_std=.1)
    assert r.mean.shape == (2, 5, 4)
    assert r.covariance.shape == (2, 5, 4, 4)
    assert bool(jnp.all(jnp.isfinite(r.mean)))


def test_particle_filter_supports_non_gaussian_noise():
    for noise in ("student_t", "gmm"):
        d = _data(noise)
        r = bootstrap_particle_filter(
            d["y"], d["x0"], d["H"], jr.PRNGKey(4), TRANSITION,
            obs_std=.1, noise_type=noise, num_particles=32,
        )
        assert r.mean.shape == (2, 5, 4)
        assert r.particles.shape == (2, 5, 32, 4)
        assert bool(jnp.all(jnp.isfinite(r.mean)))
