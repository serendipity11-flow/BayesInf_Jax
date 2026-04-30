from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260429)


def save_wiener_process() -> None:
    n_steps = 900
    horizon = 1.0
    dt = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    increments = RNG.normal(0.0, np.sqrt(dt), size=n_steps)
    path = np.concatenate([[0.0], np.cumsum(increments)])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.plot(t, path, color="#1f6f8b", linewidth=2.2)
    ax.fill_between(t, path, 0.0, color="#7cc6d6", alpha=0.22)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wiener_process.png", bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close(fig)


def save_probabilistic_time_series() -> None:
    t = np.linspace(0.0, 12.0, 220)
    mean = 0.45 * np.sin(0.85 * t) + 0.085 * t - 0.15 * np.cos(1.7 * t)
    std = 0.18 + 0.055 * t + 0.08 * (np.sin(0.55 * t - 0.4) + 1.0)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.fill_between(t, lower, upper, color="#f3b35f", alpha=0.34)
    ax.plot(t, mean, color="#9b3d12", linewidth=2.4)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "probabilistic_time_series.png",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig)


def _gradient_line(ax, points: np.ndarray, colors, linewidth: float) -> None:
    for i in range(len(points) - 1):
        ax.plot(
            points[i : i + 2, 0],
            points[i : i + 2, 1],
            color=colors[i],
            linewidth=linewidth,
            solid_capstyle="round",
        )


def _set_clean_bounds(ax, points: np.ndarray, pad_ratio: float = 0.18) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-6)
    radius = 0.5 * span.max() * (1.0 + pad_ratio)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def latent_linear_gaussian_paths() -> np.ndarray:
    n_steps = 70
    n_paths = 42
    theta = 0.34
    rho = 0.965
    F = rho * np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    sigma = 0.085
    paths = np.zeros((n_paths, n_steps, 2))
    paths[:, 0, :] = np.array([1.35, -0.22]) + RNG.normal(0.0, 0.035, size=(n_paths, 2))
    for t in range(1, n_steps):
        paths[:, t, :] = paths[:, t - 1, :] @ F.T + RNG.normal(
            0.0, sigma, size=(n_paths, 2)
        )
    return paths


def latent_mean_path(n_steps: int) -> np.ndarray:
    theta = 0.34
    rho = 0.965
    F = rho * np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    mean = np.zeros((n_steps, 2))
    mean[0] = np.array([1.35, -0.22])
    for t in range(1, n_steps):
        mean[t] = F @ mean[t - 1]
    return mean


def decode_latent_paths(paths: np.ndarray) -> np.ndarray:
    z0 = paths[..., 0]
    z1 = paths[..., 1]
    return np.stack(
        [
            1.2 * z0 + 0.26 * np.sin(2.3 * z1),
            0.82 * z1 + 0.34 * np.sin(1.8 * z0),
        ],
        axis=-1,
    )


def save_latent_space_schematic() -> None:
    paths = latent_linear_gaussian_paths()
    mean = latent_mean_path(paths.shape[1])
    highlight = paths[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(highlight)))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    for path in paths[1:]:
        ax.plot(path[:, 0], path[:, 1], color="#5aa6a6", alpha=0.11, linewidth=1.0)
    ax.plot(mean[:, 0], mean[:, 1], color="#1f2933", alpha=0.32, linewidth=2.1, linestyle="--")
    _gradient_line(ax, highlight, colors, linewidth=4.4)

    _set_clean_bounds(ax, paths.reshape(-1, 2))
    fig.tight_layout(pad=0)
    fig.savefig(OUT_DIR / "latent_space_path.png", bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close(fig)


def save_state_space_schematic() -> None:
    latent_paths = latent_linear_gaussian_paths()
    x_paths = decode_latent_paths(latent_paths)
    x_mean = decode_latent_paths(latent_mean_path(latent_paths.shape[1])[None])[0]
    highlight = x_paths[0]
    colors = plt.cm.plasma(np.linspace(0.15, 0.92, len(highlight)))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    for path in x_paths[1:]:
        ax.plot(path[:, 0], path[:, 1], color="#d18b47", alpha=0.11, linewidth=1.0)
    ax.plot(x_mean[:, 0], x_mean[:, 1], color="#1f2933", alpha=0.3, linewidth=2.1, linestyle="--")
    _gradient_line(ax, highlight, colors, linewidth=4.4)

    _set_clean_bounds(ax, x_paths.reshape(-1, 2))
    fig.tight_layout(pad=0)
    fig.savefig(OUT_DIR / "state_space_path.png", bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    save_wiener_process()
    save_probabilistic_time_series()
    save_latent_space_schematic()
    save_state_space_schematic()
