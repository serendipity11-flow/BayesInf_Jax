"""Plot unsupervised and parameter-aware maps of MEMS encoder outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems
from experiments.MEMS_end_of_line.train_inference import MODEL_NAMES, build_model


ENCODERS = ("lstm_bn_flow", "lstm_flow", "transformer_flow")
DISPLAY_NAMES = ("LSTM + BatchNorm", "LSTM without BatchNorm", "Transformer")
PARAMETER_NAMES = (r"$D_L$", r"$f_0$", r"$m$", r"$d_{off}$", "p1", "p2")


def load_encoder_output(name, signals, param_dim, config, results_dir, seed):
    key = jr.PRNGKey(seed + MODEL_NAMES.index(name) * 1000)
    model = build_model(
        name, param_dim, key, int(config.get("width", 64)),
        int(config.get("diffusion_steps", 100)),
        int(config.get("encoder_layers", 2)),
        int(config.get("transformer_heads", 4)),
        float(config.get("flow_dropout", 0.0)),
        config.get("transformer_model_dim"),
        config.get("transformer_ff_dim"),
        config.get("pretrained_lstm_bn_encoder_dir"),
    )
    model = eqx.tree_deserialise_leaves(results_dir / f"{name}.eqx", model)
    state = model.initial_batchnorm_state()
    if state is not None:
        state = eqx.tree_deserialise_leaves(
            results_dir / f"{name}.state.eqx", state)
    encoded, _ = model._encode(jnp.asarray(signals), state, inference=True)
    return np.asarray(encoded)


def scatter_feature_map(axes, coordinates_by_model, parameters, split,
                        component_names):
    """Draw a consistently styled feature map for each model and parameter."""
    markers = {"train": "o", "validation": "^", "test": "s"}
    sizes = {"train": 13, "validation": 22, "test": 22}
    for row, (name, display) in enumerate(zip(ENCODERS, DISPLAY_NAMES)):
        coordinates = coordinates_by_model[name]
        for column, parameter_name in enumerate(PARAMETER_NAMES):
            axis = axes[row, column]
            color_min = parameters[:, column].min()
            color_max = parameters[:, column].max()
            for subset in markers:
                selected = split == subset
                points = axis.scatter(
                    coordinates[selected, 0], coordinates[selected, 1],
                    c=parameters[selected, column], cmap="viridis",
                    vmin=color_min, vmax=color_max,
                    marker=markers[subset], s=sizes[subset], alpha=0.82,
                    linewidths=0, label=subset,
                )
            axis.figure.colorbar(points, ax=axis, fraction=0.046, pad=0.03)
            if row == 0:
                axis.set_title(parameter_name)
            if column == 0:
                axis.set_ylabel(f"{display}\n{component_names[1]}")
            if row == len(ENCODERS) - 1:
                axis.set_xlabel(component_names[0])


def linear_probe_scores(features, parameters, train, test):
    """Measure linearly accessible test information without test leakage."""
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(features[train], parameters[train])
    prediction = probe.predict(features[test])
    return r2_score(parameters[test], prediction, multioutput="raw_values")


def plot_gradient_maps(axes, coordinates_by_model, parameters, split):
    """Plot train-only local parameter gradients over the PLS coordinates."""
    train = split == "train"
    overlay = {"validation": "^", "test": "s"}
    for row, (name, display) in enumerate(zip(ENCODERS, DISPLAY_NAMES)):
        coordinates = coordinates_by_model[name]
        training_coordinates = coordinates[train]
        x_margin = 0.05 * np.ptp(training_coordinates[:, 0])
        y_margin = 0.05 * np.ptp(training_coordinates[:, 1])
        x_grid = np.linspace(training_coordinates[:, 0].min() - x_margin,
                             training_coordinates[:, 0].max() + x_margin, 140)
        y_grid = np.linspace(training_coordinates[:, 1].min() - y_margin,
                             training_coordinates[:, 1].max() + y_margin, 140)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid = np.column_stack((xx.ravel(), yy.ravel()))

        # Hide background predictions far away from observed training features.
        support = NearestNeighbors(n_neighbors=6).fit(training_coordinates)
        train_distances = support.kneighbors(training_coordinates)[0][:, -1]
        support_radius = 1.5 * np.quantile(train_distances, 0.9)
        grid_distance = support.kneighbors(grid, n_neighbors=1)[0][:, 0]
        supported = grid_distance <= support_radius

        for column, parameter_name in enumerate(PARAMETER_NAMES):
            axis = axes[row, column]
            color_min = parameters[:, column].min()
            color_max = parameters[:, column].max()
            smoother = KNeighborsRegressor(n_neighbors=20, weights="distance")
            smoother.fit(training_coordinates, parameters[train, column])
            gradient = np.full(grid.shape[0], np.nan)
            gradient[supported] = smoother.predict(grid[supported])
            gradient = gradient.reshape(xx.shape)
            surface = axis.contourf(
                xx, yy, gradient, levels=np.linspace(color_min, color_max, 18),
                cmap="viridis", vmin=color_min, vmax=color_max, extend="both")
            axis.scatter(training_coordinates[:, 0], training_coordinates[:, 1],
                         s=5, c="black", alpha=0.12, linewidths=0)
            for subset, marker in overlay.items():
                selected = split == subset
                axis.scatter(coordinates[selected, 0], coordinates[selected, 1],
                             marker=marker, s=22, facecolors="none",
                             edgecolors="black", linewidths=0.55, label=subset)
            axis.figure.colorbar(surface, ax=axis, fraction=0.046, pad=0.03)
            if row == 0:
                axis.set_title(parameter_name)
            if column == 0:
                axis.set_ylabel(f"{display}\nPLS2")
            if row == len(ENCODERS) - 1:
                axis.set_xlabel("PLS1")


def plot_pls_correlation_biplot(axes, coordinates_by_model, parameters, split):
    """Show how each target correlates with the two selected PLS scores."""
    train = split == "train"
    correlations = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(PARAMETER_NAMES)))
    for axis, name, display in zip(axes, ENCODERS, DISPLAY_NAMES):
        coordinates = coordinates_by_model[name]
        model_correlations = np.empty((len(PARAMETER_NAMES), 2))
        for column, (parameter_name, color) in enumerate(
                zip(PARAMETER_NAMES, colors)):
            target = parameters[train, column]
            corr_x = np.corrcoef(coordinates[train, 0], target)[0, 1]
            corr_y = np.corrcoef(coordinates[train, 1], target)[0, 1]
            model_correlations[column] = (corr_x, corr_y)
            axis.annotate(
                "", xy=(corr_x, corr_y), xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0})
            axis.text(corr_x * 1.08, corr_y * 1.08, parameter_name,
                      color=color, ha="center", va="center", fontsize=11,
                      fontweight="bold")
        correlations.append(model_correlations)
        circle = plt.Circle((0, 0), 1, fill=False, color="0.55", ls="--", lw=1)
        axis.add_patch(circle)
        axis.axhline(0, color="0.8", lw=0.7)
        axis.axvline(0, color="0.8", lw=0.7)
        axis.set_xlim(-1.25, 1.25)
        axis.set_ylim(-1.25, 1.25)
        axis.set_aspect("equal")
        axis.set_title(display)
        axis.set_xlabel(r"corr(PLS1, parameter)")
    axes[0].set_ylabel(r"corr(PLS2, parameter)")
    return np.asarray(correlations)


def plot_selected_pls_gradients(axes, coordinates_by_model, selections,
                                parameters, split):
    """Plot each target using its best-correlated pair among six PLS scores."""
    train = split == "train"
    for row, (name, display) in enumerate(zip(ENCODERS, DISPLAY_NAMES)):
        all_coordinates = coordinates_by_model[name]
        for column, parameter_name in enumerate(PARAMETER_NAMES):
            first, second = selections[name][column]
            coordinates = all_coordinates[:, [first, second]]
            training_coordinates = coordinates[train]
            x_margin = 0.05 * np.ptp(training_coordinates[:, 0])
            y_margin = 0.05 * np.ptp(training_coordinates[:, 1])
            xx, yy = np.meshgrid(
                np.linspace(training_coordinates[:, 0].min() - x_margin,
                            training_coordinates[:, 0].max() + x_margin, 140),
                np.linspace(training_coordinates[:, 1].min() - y_margin,
                            training_coordinates[:, 1].max() + y_margin, 140))
            grid = np.column_stack((xx.ravel(), yy.ravel()))
            support = NearestNeighbors(n_neighbors=6).fit(training_coordinates)
            local_scale = support.kneighbors(training_coordinates)[0][:, -1]
            radius = 1.5 * np.quantile(local_scale, 0.9)
            supported = support.kneighbors(grid, n_neighbors=1)[0][:, 0] <= radius
            smoother = KNeighborsRegressor(n_neighbors=20, weights="distance")
            smoother.fit(training_coordinates, parameters[train, column])
            gradient = np.full(len(grid), np.nan)
            gradient[supported] = smoother.predict(grid[supported])
            gradient = gradient.reshape(xx.shape)
            color_min = parameters[:, column].min()
            color_max = parameters[:, column].max()
            axis = axes[row, column]
            surface = axis.contourf(
                xx, yy, gradient, levels=np.linspace(color_min, color_max, 18),
                cmap="viridis", vmin=color_min, vmax=color_max, extend="both")
            axis.scatter(training_coordinates[:, 0], training_coordinates[:, 1],
                         s=5, c="black", alpha=0.12, linewidths=0)
            for subset, marker in (("validation", "^"), ("test", "s")):
                selected = split == subset
                axis.scatter(coordinates[selected, 0], coordinates[selected, 1],
                             marker=marker, s=22, facecolors="none",
                             edgecolors="black", linewidths=0.55, label=subset)
            axis.figure.colorbar(surface, ax=axis, fraction=0.046, pad=0.03)
            axis.set_xlabel(f"PLS{first + 1}")
            axis.set_ylabel(f"PLS{second + 1}" if column == 0 else "")
            if row == 0:
                axis.set_title(parameter_name)
            if column == 0:
                axis.set_ylabel(f"{display}\nPLS{second + 1}")


def plot_parameter_specific_pls(axes, scores, parameters, split):
    """Plot one independently supervised PLS score for each target."""
    styles = {"train": ("o", 13), "validation": ("^", 24), "test": ("s", 24)}
    for row, (name, display) in enumerate(zip(ENCODERS, DISPLAY_NAMES)):
        for column, parameter_name in enumerate(PARAMETER_NAMES):
            axis = axes[row, column]
            score = scores[name][:, column]
            for subset, (marker, size) in styles.items():
                selected = split == subset
                axis.scatter(score[selected], parameters[selected, column],
                             marker=marker, s=size, alpha=0.75, linewidths=0,
                             label=subset)
            if row == 0:
                axis.set_title(parameter_name)
            if column == 0:
                axis.set_ylabel(f"{display}\nparameter value")
            if row == len(ENCODERS) - 1:
                axis.set_xlabel("parameter-specific PLS1")


def main():
    global ENCODERS, DISPLAY_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=ENCODERS,
                        default=list(ENCODERS))
    args = parser.parse_args()
    display_by_name = dict(zip(ENCODERS, DISPLAY_NAMES))
    ENCODERS = tuple(args.models)
    DISPLAY_NAMES = tuple(display_by_name[name] for name in ENCODERS)
    feature_dir = args.results_dir / "feature_maps"
    feature_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.results_dir / "config.json").read_text())
    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    data = load_mems(
        args.data_dir, seed=seed,
        standardize_signals=not config.get("no_signal_standardization", False),
    )
    signals = np.concatenate((data.train_x, data.val_x, data.test_x))
    parameters = np.concatenate((data.train_theta, data.val_theta, data.test_theta))
    split = np.repeat(("train", "validation", "test"), (200, 50, 50))

    row_count = len(ENCODERS)
    figure_height = 3.2 * row_count + 1
    fig, axes = plt.subplots(row_count, 6, figsize=(21, figure_height),
                             sharex="row", sharey="row", squeeze=False)
    variance_rows, archive = [], {"parameters": parameters, "split": split}
    pca_coordinates, pls_coordinates, pls6_coordinates = {}, {}, {}
    pls6_selections, specific_pls_scores = {}, {}
    probe_2pc, probe_full = [], []
    train, test = split == "train", split == "test"
    for row, (name, display) in enumerate(zip(ENCODERS, DISPLAY_NAMES)):
        hidden = load_encoder_output(
            name, signals, parameters.shape[1], config, args.results_dir, seed)
        pca = PCA(n_components=2).fit(hidden)
        coordinates = pca.transform(hidden)
        pca_coordinates[name] = coordinates
        pls = PLSRegression(n_components=2, scale=True)
        pls.fit(hidden[train], parameters[train])
        pls_coordinates[name] = pls.transform(hidden)
        component_count = min(6, hidden.shape[1], int(train.sum()) - 1)
        pls6 = PLSRegression(n_components=component_count, scale=True)
        pls6.fit(hidden[train], parameters[train])
        pls6_coordinates[name] = pls6.transform(hidden)
        target_selections = []
        for column in range(parameters.shape[1]):
            correlations = np.array([
                np.corrcoef(pls6_coordinates[name][train, component],
                            parameters[train, column])[0, 1]
                for component in range(component_count)
            ])
            target_selections.append(
                np.argsort(np.abs(correlations))[-2:][::-1])
        pls6_selections[name] = np.asarray(target_selections)

        target_scores = []
        for column in range(parameters.shape[1]):
            target_pls = PLSRegression(n_components=1, scale=True)
            target_pls.fit(hidden[train], parameters[train, column])
            target_scores.append(target_pls.transform(hidden)[:, 0])
        specific_pls_scores[name] = np.column_stack(target_scores)
        probe_2pc.append(linear_probe_scores(
            coordinates, parameters, train, test))
        probe_full.append(linear_probe_scores(
            hidden, parameters, train, test))
        archive[f"{name}_hidden"] = hidden
        archive[f"{name}_pca"] = coordinates
        archive[f"{name}_pls"] = pls_coordinates[name]
        archive[f"{name}_pls6"] = pls6_coordinates[name]
        archive[f"{name}_pls6_selections"] = pls6_selections[name]
        archive[f"{name}_parameter_specific_pls"] = specific_pls_scores[name]
        archive[f"{name}_explained_variance_ratio"] = pca.explained_variance_ratio_
        variance_rows.append((name, *pca.explained_variance_ratio_,
                              pca.explained_variance_ratio_.sum()))
    scatter_feature_map(axes, pca_coordinates, parameters, split, ("PC1", "PC2"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.suptitle("PCA of raw-signal MEMS encoder outputs", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(feature_dir / "encoder_pca_feature_maps.png", dpi=200)
    fig.savefig(feature_dir / "encoder_pca_feature_maps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(row_count, 6, figsize=(21, figure_height),
                             sharex="row", sharey="row", squeeze=False)
    scatter_feature_map(axes, pls_coordinates, parameters, split,
                        ("PLS1", "PLS2"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.suptitle("Parameter-aware PLS maps of MEMS encoder outputs (train fit)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(feature_dir / "encoder_pls_feature_maps.png", dpi=200)
    fig.savefig(feature_dir / "encoder_pls_feature_maps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(row_count, 6, figsize=(21, figure_height),
                             sharex="row", sharey="row", squeeze=False)
    plot_gradient_maps(axes, pls_coordinates, parameters, split)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    fig.suptitle(
        "Train-only local parameter gradients in PLS feature space", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(feature_dir / "encoder_pls_gradient_maps.png", dpi=200)
    fig.savefig(feature_dir / "encoder_pls_gradient_maps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(row_count, 6, figsize=(21, figure_height),
                             squeeze=False)
    plot_selected_pls_gradients(
        axes, pls6_coordinates, pls6_selections, parameters, split)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    fig.suptitle(
        "Six-component PLS: best two components selected per parameter", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(feature_dir / "encoder_pls6_selected_gradient_maps.png", dpi=200)
    fig.savefig(feature_dir / "encoder_pls6_selected_gradient_maps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(row_count, 6, figsize=(18, figure_height),
                             sharey="col", squeeze=False)
    plot_parameter_specific_pls(axes, specific_pls_scores, parameters, split)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.suptitle(
        "Parameter-specific one-dimensional PLS diagnostics (train fit)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(feature_dir / "encoder_parameter_specific_pls.png", dpi=200)
    fig.savefig(feature_dir / "encoder_parameter_specific_pls.pdf")
    plt.close(fig)

    fig, axes_grid = plt.subplots(
        1, row_count, figsize=(4.7 * row_count, 4.8), sharex=True, sharey=True,
        squeeze=False, layout="constrained")
    axes = axes_grid[0]
    pls_target_correlations = plot_pls_correlation_biplot(
        axes, pls_coordinates, parameters, split)
    fig.suptitle(
        "Why PLS1 and PLS2 are selected: correlation with training targets")
    fig.savefig(feature_dir / "encoder_pls_dimension_selection.png", dpi=200)
    fig.savefig(feature_dir / "encoder_pls_dimension_selection.pdf")
    plt.close(fig)
    archive["pls_target_correlations"] = pls_target_correlations

    probe_2pc, probe_full = np.asarray(probe_2pc), np.asarray(probe_full)
    archive["linear_probe_r2_2pc"] = probe_2pc
    archive["linear_probe_r2_full"] = probe_full
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 4.2), sharey=True, layout="constrained")
    for axis, scores, title in zip(
            axes, (probe_2pc, probe_full),
            ("First two PCs", "Full encoder output")):
        image = axis.imshow(scores, cmap="RdYlBu", vmin=-0.1, vmax=1.0,
                            aspect="auto")
        axis.set_xticks(range(len(PARAMETER_NAMES)), PARAMETER_NAMES)
        axis.set_yticks(range(len(DISPLAY_NAMES)), DISPLAY_NAMES)
        axis.set_title(f"{title}: test linear-probe $R^2$")
        for row in range(scores.shape[0]):
            for column in range(scores.shape[1]):
                value = scores[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          color="white" if value < 0.15 else "black", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="$R^2$")
    fig.savefig(feature_dir / "encoder_feature_information.png", dpi=200)
    fig.savefig(feature_dir / "encoder_feature_information.pdf")
    plt.close(fig)

    np.savez_compressed(feature_dir / "encoder_pca_features.npz", **archive)
    with (feature_dir / "encoder_pca_explained_variance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "pc1", "pc2", "pc1_plus_pc2"))
        writer.writerows(variance_rows)
    with (feature_dir / "encoder_linear_probe_r2.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "representation", *PARAMETER_NAMES))
        for name, scores_2pc, scores_full in zip(
                ENCODERS, probe_2pc, probe_full):
            writer.writerow((name, "first_2_pcs", *scores_2pc))
            writer.writerow((name, "full_encoder_output", *scores_full))
    with (feature_dir / "encoder_pls_target_correlations.csv").open(
            "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "parameter", "corr_pls1", "corr_pls2"))
        for name, model_correlations in zip(ENCODERS, pls_target_correlations):
            for parameter_name, values in zip(PARAMETER_NAMES, model_correlations):
                writer.writerow((name, parameter_name, *values))
    with (feature_dir / "encoder_pls6_component_selection.csv").open(
            "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "parameter", "first_component",
                         "second_component", "corr_first", "corr_second"))
        for name in ENCODERS:
            coordinates = pls6_coordinates[name]
            for column, parameter_name in enumerate(PARAMETER_NAMES):
                first, second = pls6_selections[name][column]
                corr_first = np.corrcoef(
                    coordinates[train, first], parameters[train, column])[0, 1]
                corr_second = np.corrcoef(
                    coordinates[train, second], parameters[train, column])[0, 1]
                writer.writerow((name, parameter_name, first + 1, second + 1,
                                 corr_first, corr_second))
    for row in variance_rows:
        print(f"{row[0]}: PC1={row[1]:.4f}, PC2={row[2]:.4f}, total={row[3]:.4f}")


if __name__ == "__main__":
    main()
