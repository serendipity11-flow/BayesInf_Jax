from __future__ import annotations

import numpy as np


def energy_score(samples, truth):
    first = np.linalg.norm(samples - truth[:, None, :], axis=-1).mean(axis=1)
    # Independent cyclic pairing avoids the quadratic sample-memory cost.
    second = np.linalg.norm(samples - np.roll(samples, 1, axis=1), axis=-1).mean(axis=1)
    return float(np.mean(first - 0.5 * second))


def sliced_wasserstein(a, b, seed=0, projections=128):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, a.shape[-1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = []
    for direction in directions:
        pa = np.sort(a @ direction, axis=1)
        pb = np.sort(b @ direction, axis=1)
        distances.append(np.mean(np.abs(pa - pb)))
    return float(np.mean(distances))


def summarize(samples, truth, case):
    mean = samples.mean(axis=1)
    error = mean - truth
    result = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "energy_score": energy_score(samples, truth),
    }
    for level in (0.5, 0.8, 0.9, 0.95):
        tail = (1.0 - level) / 2.0
        lower, upper = np.quantile(samples, (tail, 1.0 - tail), axis=1)
        result[f"coverage_{int(level * 100)}"] = float(np.mean((truth >= lower) & (truth <= upper)))
        result[f"width_{int(level * 100)}"] = float(np.mean(upper - lower))
    if case == "linear":
        ratio = samples[..., 1] / samples[..., 0]
        result["ratio_k_over_m_mae"] = float(np.mean(np.abs(ratio.mean(1) - truth[:, 1] / truth[:, 0])))
        result["constraint_k_minus_10m"] = float(np.mean(np.abs(samples[..., 1] - 10.0 * samples[..., 0])))
    if case == "multimodal":
        # With the symmetric observation, the two labelings are equivalent.
        direct = np.linalg.norm(samples - truth[:, None], axis=-1)
        swapped = np.linalg.norm(samples - truth[:, None, ::-1], axis=-1)
        result["swap_invariant_error"] = float(np.minimum(direct, swapped).mean())
        result["mode_weight_error"] = float(np.abs((samples[..., 0] < samples[..., 1]).mean(axis=1) - 0.5).mean())
    return result
