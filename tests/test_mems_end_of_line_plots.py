import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.MEMS_end_of_line.plots import plot_parameter_regression


def test_six_parameter_regression_plot(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(8, 6))
    samples = truth[None] + rng.normal(scale=0.2, size=(10, 8, 6))
    output = tmp_path / "regression.png"
    plot_parameter_regression(samples, truth, "fm", output)
    assert output.exists()
    assert output.stat().st_size > 1_000
