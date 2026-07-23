"""Compare linear and multimodal parameter-identification checkpoints.

The script evaluates the seven amortised posterior estimators used by
``cross_evaluation_and_plot_mm_l.py`` and writes:

* ``parameter_identification_linear_vs_multimodal_metrics.csv``
* ``parameter_identification_linear_vs_multimodal_metrics.pdf``
* ``parameter_identification_linear_vs_multimodal_posteriors.pdf``

Multimodal metrics are computed in the identifiable (unordered) parameter
space by sorting (k1, k3).  Thus swapping k1 and k3 is not counted as an
error, while the posterior plot still shows both symmetric modes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from functools import partial

os.environ.setdefault("MPLBACKEND", "Agg")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bayesinf.utils.generate_data import (
    denormalize_data,
    generate_mass_spring_data2,
    generate_mass_spring_data_l,
    normalize_data,
)


FLOW_TYPES = ["BNN", "VAE", "DDPM", "SBM", "FM", "RealNVP", "NSF"]
COLORS = dict(zip(FLOW_TYPES, sns.color_palette("colorblind", len(FLOW_TYPES))))


def create_model(flow_type: str, param_dim: int, key: jax.Array):
    feature_dim, latent_dim, hidden_rnn = 2, 64, 64
    hidden_sizes, hidden_time = [64, 64], 4
    name = flow_type.upper()
    if name == "BNN":
        from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator
        return BNNPosteriorEstimator(feature_dim, param_dim, latent_dim, hidden_sizes, hidden_rnn, key=key)
    if name == "VAE":
        from bayesinf.models.variational_autoencoder import cVAE
        return cVAE(feature_dim, param_dim, latent_dim, hidden_sizes, hidden_rnn, key=key)
    if name == "DDPM":
        from bayesinf.models.denoising_diffusion_model import ConditionalDDPM
        return ConditionalDDPM(feature_dim, param_dim, latent_dim, hidden_rnn, hidden_time, hidden_sizes, key=key)
    if name == "SBM":
        from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel
        return ConditionalScoreModel(feature_dim, param_dim, latent_dim, hidden_rnn, hidden_time, hidden_sizes, key=key)
    if name == "FM":
        from bayesinf.models.flow_matching import ConditionalFlowMatching
        return ConditionalFlowMatching(feature_dim, param_dim, latent_dim, hidden_sizes, hidden_rnn, key=key)

    from bayesinf.models.normalizing_flow import CondNormFlow
    return CondNormFlow(
        param_dim, feature_dim, key=key, n_coupling=8,
        hidden_nf=hidden_sizes[0], num_hidden_nf=len(hidden_sizes),
        encoded_latent_dim=latent_dim, hidden_rnn=hidden_rnn,
        rnn_type="lstm", flow_type=flow_type,
    )


def prepare_dataset(kind: str, test_size: int):
    generator = generate_mass_spring_data_l if kind == "linear" else generate_mass_spring_data2
    generator = partial(jax.jit, static_argnames=["n_samples", "dt0", "t1", "noise_std"])(generator)
    observations, truth = generator(jr.PRNGKey(0), dt0=0.05, t1=1.0, n_samples=test_size, noise_std=0.0)
    observations, _, _ = normalize_data(observations)
    _, theta_mean, theta_std = normalize_data(truth)
    return observations, truth, theta_mean, theta_std


def metrics(samples: np.ndarray, truth: np.ndarray, multimodal: bool):
    # In the multimodal system the likelihood identifies {k1, k3}, not their
    # labels. Sorting maps both symmetric modes into that quotient space.
    if multimodal:
        samples = np.sort(samples, axis=-1)
        truth = np.sort(truth, axis=-1)
    posterior_mean = samples.mean(axis=0)
    rmse_per_parameter = np.sqrt(np.mean((posterior_mean - truth) ** 2, axis=0))
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    mciw_per_parameter = np.mean(high - low, axis=0)
    cicp_per_parameter = np.mean((truth >= low) & (truth <= high), axis=0)
    return rmse_per_parameter, mciw_per_parameter, cicp_per_parameter


def evaluate(kind: str, args):
    observations, truth, theta_mean, theta_std = prepare_dataset(kind, args.test_size)
    param_dim = truth.shape[-1]
    dataset = f"mass_spring_damper_0.05_1.0_{kind if kind == 'linear' else 'multi_modal'}"
    records, sample_cache = [], {}

    for index, flow in enumerate(FLOW_TYPES):
        path = os.path.join(args.results_dir, f"{flow}_{dataset}.pt")
        if not os.path.exists(path):
            print(f"Skipping missing checkpoint: {path}", file=sys.stderr)
            continue
        template = create_model(flow, param_dim, jr.PRNGKey(42))
        model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(path, template))

        @eqx.filter_jit
        def sample(d, key):
            return model.sample(d, key, args.posterior_samples)

        # Compile the exact full-batch operation before measuring execution.
        timed_key = jr.fold_in(jr.PRNGKey(123), index)
        sample(observations, timed_key).block_until_ready()
        timings = []
        raw = None
        for repeat in range(args.timing_repeats):
            key = jr.fold_in(timed_key, repeat + 1)
            start = time.perf_counter()
            raw = sample(observations, key).block_until_ready()
            timings.append(time.perf_counter() - start)
        assert raw is not None
        samples = np.asarray(denormalize_data(raw, theta_mean[None], theta_std[None]))
        truth_np = np.asarray(truth)
        rmse, mciw, cicp = metrics(samples, truth_np, kind == "multi_modal")
        sample_cache[flow] = samples
        records.append({
            "dataset": kind,
            "model": flow,
            "rmse": float(rmse.mean()),
            "mciw": float(mciw.mean()),
            "cicp": float(cicp.mean()),
            "inference_time_ms": 1000.0 * float(np.median(timings)) / args.test_size,
            "rmse_per_parameter": rmse,
            "mciw_per_parameter": mciw,
            "cicp_per_parameter": cicp,
        })
        print(f"{kind:11s} {flow:7s} RMSE={rmse.mean():.4g} MCIW={mciw.mean():.4g} "
              f"CICP={cicp.mean():.3f} time={records[-1]['inference_time_ms']:.3f} ms/case")
    return records, sample_cache, np.asarray(truth)


def write_csv(records, path: str):
    fields = ["dataset", "model", "parameter", "rmse", "mciw", "cicp", "inference_time_ms"]
    parameter_names = {"linear": ["m", "k", "c"], "multi_modal": ["min(k1,k3)", "max(k1,k3)"]}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields if key != "parameter"} | {"parameter": "mean"})
            for i, parameter in enumerate(parameter_names[record["dataset"]]):
                writer.writerow({
                    "dataset": record["dataset"], "model": record["model"], "parameter": parameter,
                    "rmse": record["rmse_per_parameter"][i], "mciw": record["mciw_per_parameter"][i],
                    "cicp": record["cicp_per_parameter"][i],
                    "inference_time_ms": record["inference_time_ms"],
                })


def plot_metrics(records, path: str):
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    specifications = [
        ("rmse", "RMSE (lower is better)"), ("mciw", "MCIW (lower is sharper)"),
        ("cicp", "CICP (target: 0.95)"), ("inference_time_ms", "Inference time (ms/case)"),
    ]
    x = np.arange(len(FLOW_TYPES)); width = 0.38
    for ax, (key, label) in zip(axes, specifications):
        for offset, (dataset, hatch) in zip((-width / 2, width / 2), (("linear", ""), ("multi_modal", "//"))):
            lookup = {r["model"]: r[key] for r in records if r["dataset"] == dataset}
            values = [lookup.get(model, np.nan) for model in FLOW_TYPES]
            ax.bar(x + offset, values, width, label=dataset.replace("_", " ").title(),
                   color=[COLORS[m] for m in FLOW_TYPES], edgecolor="black", linewidth=.6, hatch=hatch)
        ax.set_ylabel(label); ax.set_xticks(x, FLOW_TYPES, rotation=35, ha="right"); ax.grid(axis="y", ls=":", alpha=.6)
        if key == "cicp":
            ax.axhline(.95, color="black", ls="--", lw=1.5)
            ax.set_ylim(0, 1.05)
        elif key in {"rmse", "mciw", "inference_time_ms"}:
            positive = [r[key] for r in records if r[key] > 0]
            if positive and max(positive) / min(positive) > 20:
                ax.set_yscale("log")
    axes[0].legend(frameon=True)
    fig.suptitle("Linear vs multimodal parameter identification", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posteriors(caches, truths, path: str, case_index: int):
    """One representative identification plot per dataset and model."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, len(FLOW_TYPES), figsize=(4.2 * len(FLOW_TYPES), 8.5))
    linear_truth, mm_truth = truths["linear"][case_index], truths["multi_modal"][case_index]
    for col, flow in enumerate(FLOW_TYPES):
        ax = axes[0, col]
        if flow in caches["linear"]:
            s = caches["linear"][flow][:, case_index]
            ax.scatter(s[:, 0], s[:, 1], s=12, alpha=.25, color=COLORS[flow])
            ax.scatter(linear_truth[0], linear_truth[1], marker="*", s=240, color="gold", edgecolor="black", zorder=5)
        ax.set_title(flow, fontweight="bold"); ax.set_xlabel("m");
        if col == 0: ax.set_ylabel("Linear\nk")

        ax = axes[1, col]
        if flow in caches["multi_modal"]:
            s = caches["multi_modal"][flow][:, case_index]
            if np.std(s[:, 0]) > 0 and np.std(s[:, 1]) > 0:
                sns.kdeplot(x=s[:, 0], y=s[:, 1], ax=ax, fill=True, levels=7, cmap="Blues", thresh=.05)
            ax.scatter(s[:, 0], s[:, 1], s=10, alpha=.22, color=COLORS[flow])
            ax.scatter(*mm_truth, marker="X", s=180, color="orange", edgecolor="black", zorder=5)
            ax.scatter(mm_truth[1], mm_truth[0], marker="X", s=180, color="green", edgecolor="black", zorder=5)
        ax.set_xlabel(r"$k_1$")
        if col == 0: ax.set_ylabel("Multimodal\n" + r"$k_3$")
    fig.suptitle(f"Representative posterior identification (test case {case_index})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "results_parameter_identification"))
    parser.add_argument("--output-dir", default=CURRENT_DIR)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--posterior-samples", type=int, default=100)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--case-index", type=int, default=49)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.case_index < args.test_size:
        raise ValueError("--case-index must be smaller than --test-size")
    os.makedirs(args.output_dir, exist_ok=True)
    all_records, caches, truths = [], {}, {}
    for kind in ("linear", "multi_modal"):
        records, caches[kind], truths[kind] = evaluate(kind, args)
        all_records.extend(records)
    stem = os.path.join(args.output_dir, "parameter_identification_linear_vs_multimodal")
    write_csv(all_records, stem + "_metrics.csv")
    plot_metrics(all_records, stem + "_metrics.pdf")
    plot_posteriors(caches, truths, stem + "_posteriors.pdf", args.case_index)
    print(f"Wrote {stem}_metrics.csv, {stem}_metrics.pdf, and {stem}_posteriors.pdf")


if __name__ == "__main__":
    main()
