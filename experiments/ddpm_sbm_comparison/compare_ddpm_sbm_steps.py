"""Evaluate DDPM and SBM parameter identification at different sampling steps.

Produces four separate metric/posterior figures (one per model and dataset)
and a CSV containing all aggregate and per-parameter measurements.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from compare_parameter_identification_linear_multimodal import (
    CURRENT_DIR,
    PROJECT_ROOT,
    create_model,
    metrics,
    prepare_dataset,
)
from bayesinf.utils.generate_data import denormalize_data


MODELS = ("DDPM", "SBM")


def sample_ddim(model, observations, key, num_samples: int, steps: int):
    """Deterministic DDIM sampling on a subset of the trained DDPM schedule."""
    batch_size = observations.shape[0]
    encoded = model.encoder(observations)
    condition = jnp.repeat(encoded, num_samples, axis=0)
    unconditional = jnp.zeros_like(condition)
    total = batch_size * num_samples
    key_initial, key_noise = jr.split(key)
    z = jr.normal(key_initial, (total, model.param_dim))
    # Descending, unique trained indices; step=1000 uses the complete schedule.
    timesteps = jnp.round(jnp.linspace(model.n_steps - 1, 0, steps)).astype(jnp.int32)
    noises = jr.normal(key_noise, (steps, total, model.param_dim))

    def step(z_t, values):
        position, timestep, noise = values
        time_embedding = model.time_emb(timestep.astype(jnp.float32))
        time_embedding = jnp.broadcast_to(time_embedding, (total, time_embedding.shape[-1]))
        noise_conditional = jax.vmap(model.net)(z_t, time_embedding, condition)
        noise_unconditional = jax.vmap(model.net)(z_t, time_embedding, unconditional)
        # Match ConditionalDDPM.sample's default classifier-free guidance.
        predicted_noise = noise_unconditional + 2.0 * (noise_conditional - noise_unconditional)
        alpha_bar = model.alphas_cumprod[timestep]
        predicted_x0 = (z_t - jnp.sqrt(1.0 - alpha_bar) * predicted_noise) / jnp.sqrt(alpha_bar)
        previous_timestep = jnp.where(position + 1 < steps, timesteps[position + 1], -1)
        previous_alpha_bar = jnp.where(
            previous_timestep >= 0, model.alphas_cumprod[jnp.maximum(previous_timestep, 0)], 1.0
        )
        # Generalised DDIM update with eta=1. At 1000 steps this recovers the
        # stochastic DDPM chain; subsampling uses the same trained schedule.
        sigma = jnp.sqrt(
            jnp.maximum((1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
                        * (1.0 - alpha_bar / previous_alpha_bar), 0.0)
        )
        direction_scale = jnp.sqrt(jnp.maximum(1.0 - previous_alpha_bar - sigma**2, 0.0))
        z_previous = (jnp.sqrt(previous_alpha_bar) * predicted_x0
                      + direction_scale * predicted_noise + sigma * noise)
        return z_previous, None

    z, _ = jax.lax.scan(step, z, (jnp.arange(steps), timesteps, noises))
    return jnp.swapaxes(z.reshape(batch_size, num_samples, model.param_dim), 0, 1)


def evaluate_model_dataset(model_name: str, dataset: str, args):
    observations, truth, theta_mean, theta_std = prepare_dataset(dataset, args.test_size)
    checkpoint_dataset = dataset if dataset == "linear" else "multi_modal"
    checkpoint = os.path.join(
        args.results_dir, f"{model_name}_mass_spring_damper_0.05_1.0_{checkpoint_dataset}.pt"
    )
    template = create_model(model_name, truth.shape[-1], jr.PRNGKey(42))
    model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(checkpoint, template))
    records, sample_cache = [], {}

    for step_count in args.steps:
        if model_name == "DDPM":
            def run(d, key):
                return sample_ddim(model, d, key, args.posterior_samples, step_count)
        else:
            def run(d, key):
                return model.sample(d, key, args.posterior_samples, steps=step_count)
        run = eqx.filter_jit(run)
        base_key = jr.fold_in(jr.PRNGKey(123), step_count)
        run(observations, base_key).block_until_ready()  # exclude compilation
        elapsed, raw = [], None
        for repeat in range(args.timing_repeats):
            start = time.perf_counter()
            raw = run(observations, jr.fold_in(base_key, repeat + 1)).block_until_ready()
            elapsed.append(time.perf_counter() - start)
        samples = np.asarray(denormalize_data(raw, theta_mean[None], theta_std[None]))
        rmse, mciw, cicp = metrics(samples, np.asarray(truth), dataset == "multi_modal")
        sample_cache[step_count] = samples
        record = {
            "dataset": dataset, "model": model_name, "steps": step_count,
            "rmse": float(rmse.mean()), "mciw": float(mciw.mean()),
            "cicp": float(cicp.mean()),
            "inference_time_ms": 1000.0 * float(np.median(elapsed)) / args.test_size,
            "rmse_per_parameter": rmse, "mciw_per_parameter": mciw,
            "cicp_per_parameter": cicp,
        }
        records.append(record)
        print(f"{model_name:4s} {dataset:11s} {step_count:4d} steps | "
              f"RMSE={record['rmse']:.4g} MCIW={record['mciw']:.4g} "
              f"CICP={record['cicp']:.3f} time={record['inference_time_ms']:.3f} ms/case",
              flush=True)
    return records, sample_cache, np.asarray(truth)


def plot_metrics(records, model_name: str, dataset: str, path: str):
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.8))
    specs = (("rmse", "RMSE", None), ("mciw", "MCIW", None),
             ("cicp", "CICP", (0, 1.05)),
             ("inference_time_ms", "Inference time (ms/case)", None))
    steps = np.asarray([r["steps"] for r in records])
    for ax, (key, ylabel, ylim) in zip(axes, specs):
        values = [r[key] for r in records]
        ax.plot(steps, values, "o-", lw=2.5, ms=7, color="#0072B2")
        ax.set_xscale("log"); ax.set_xlabel("Sampling steps"); ax.set_ylabel(ylabel)
        ax.set_xticks(steps, [str(s) for s in steps], rotation=35)
        ax.grid(ls=":", alpha=.65)
        if ylim: ax.set_ylim(*ylim)
        if key == "cicp": ax.axhline(.95, color="black", ls="--", lw=1.5, label="95% target")
        positive = [value for value in values if value > 0]
        if key != "cicp" and positive and max(positive) / min(positive) > 20:
            ax.set_yscale("log")
    axes[2].legend(fontsize=12)
    fig.suptitle(f"{model_name}: {dataset.replace('_', ' ')} parameter identification", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_posteriors(cache, truth, model_name: str, dataset: str, case_index: int, path: str):
    sns.set_theme(style="whitegrid", context="talk")
    step_counts = list(cache)
    fig, axes = plt.subplots(1, len(step_counts), figsize=(4.2 * len(step_counts), 4.5), squeeze=False)
    for ax, step_count in zip(axes[0], step_counts):
        samples = cache[step_count][:, case_index]
        if dataset == "multi_modal":
            if np.all(np.std(samples, axis=0) > 0):
                sns.kdeplot(x=samples[:, 0], y=samples[:, 1], ax=ax, fill=True,
                            levels=7, cmap="Blues", thresh=.05)
            ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=.22, color="#0072B2")
            ax.scatter(*truth[case_index], marker="X", s=180, color="orange", edgecolor="black", zorder=5)
            ax.scatter(truth[case_index, 1], truth[case_index, 0], marker="X", s=180,
                       color="green", edgecolor="black", zorder=5)
            ax.set_xlabel(r"$k_1$"); ax.set_ylabel(r"$k_3$")
        else:
            ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=.28, color="#0072B2")
            ax.scatter(truth[case_index, 0], truth[case_index, 1], marker="*", s=240,
                       color="gold", edgecolor="black", zorder=5)
            ax.set_xlabel("m"); ax.set_ylabel("k")
        ax.set_title(f"{step_count} steps", fontweight="bold")
    fig.suptitle(f"{model_name} {dataset.replace('_', ' ')} posterior — test case {case_index}", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def write_csv(records, path: str):
    fields = ["dataset", "model", "steps", "parameter", "rmse", "mciw", "cicp", "inference_time_ms"]
    names = {"linear": ("m", "k", "c"), "multi_modal": ("min(k1,k3)", "max(k1,k3)")}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            common = {key: r[key] for key in ("dataset", "model", "steps", "inference_time_ms")}
            writer.writerow(common | {"parameter": "mean", "rmse": r["rmse"], "mciw": r["mciw"], "cicp": r["cicp"]})
            for i, name in enumerate(names[r["dataset"]]):
                writer.writerow(common | {"parameter": name, "rmse": r["rmse_per_parameter"][i],
                                           "mciw": r["mciw_per_parameter"][i],
                                           "cicp": r["cicp_per_parameter"][i]})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50, 100, 250, 500, 1000])
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--posterior-samples", type=int, default=100)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--case-index", type=int, default=49)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "results_parameter_identification"))
    parser.add_argument("--output-dir", default=CURRENT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(step < 2 or step > 1000 for step in args.steps):
        raise ValueError("all --steps must be between 2 and the trained DDPM schedule length (1000)")
    if not 0 <= args.case_index < args.test_size:
        raise ValueError("--case-index must be smaller than --test-size")
    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []
    for model_name in MODELS:
        for dataset in ("linear", "multi_modal"):
            records, cache, truth = evaluate_model_dataset(model_name, dataset, args)
            all_records.extend(records)
            stem = os.path.join(args.output_dir, f"{model_name.lower()}_{dataset}_step_comparison")
            plot_metrics(records, model_name, dataset, stem + "_metrics.pdf")
            plot_posteriors(cache, truth, model_name, dataset, args.case_index, stem + "_posteriors.pdf")
    write_csv(all_records, os.path.join(args.output_dir, "ddpm_sbm_step_comparison_metrics.csv"))


if __name__ == "__main__":
    main()
