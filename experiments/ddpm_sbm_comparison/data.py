from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from bayesinf.utils.generate_data import (
    generate_mass_spring_data,
    generate_mass_spring_data2,
    generate_mass_spring_data_l,
)


@dataclass
class Dataset:
    train_observations: jax.Array
    train_parameters: jax.Array
    validation_observations: jax.Array
    validation_parameters: jax.Array
    test_observations: jax.Array
    test_parameters: jax.Array
    observation_mean: jax.Array
    observation_std: jax.Array
    parameter_mean: jax.Array
    parameter_std: jax.Array
    parameter_names: tuple[str, ...]


GENERATORS = {
    "basic": (generate_mass_spring_data, ("m", "k", "c")),
    "linear": (generate_mass_spring_data_l, ("m", "k", "c")),
    "multimodal": (generate_mass_spring_data2, ("k1", "k3")),
}


def make_dataset(case, keys, train_size, validation_size, test_size, dt, duration, noise_std):
    generator, names = GENERATORS[case]
    generate = jax.jit(
        partial(generator, dt0=dt, t1=duration, noise_std=noise_std),
        static_argnames=("n_samples",),
    )
    train_x, train_y = generate(keys[0], n_samples=train_size)
    val_x, val_y = generate(keys[1], n_samples=validation_size)
    test_x, test_y = generate(keys[2], n_samples=test_size)
    x_mean = jnp.mean(train_x, axis=(0, 1), keepdims=True)
    x_std = jnp.maximum(jnp.std(train_x, axis=(0, 1), keepdims=True), 1e-6)
    y_mean = jnp.mean(train_y, axis=0, keepdims=True)
    y_std = jnp.maximum(jnp.std(train_y, axis=0, keepdims=True), 1e-6)
    norm_x = lambda x: (x - x_mean) / x_std
    norm_y = lambda y: (y - y_mean) / y_std
    return Dataset(
        norm_x(train_x), norm_y(train_y), norm_x(val_x), norm_y(val_y), norm_x(test_x),
        test_y, x_mean, x_std, y_mean, y_std, names,
    )
