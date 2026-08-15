# MEMS end-of-line parameter identification

This experiment learns the posterior over six MEMS parameters (`theta`) from the
`sig_c1` time series. It uses a deterministic shuffled split of 200 training, 50
validation, and 50 test examples. All normalization statistics are training-only.
Every condition network is an LSTM; the compared posterior heads are BNN, cVAE,
RealNVP, NSF, flow matching (FM), DDPM, and score-based modeling (SBM).

Run the full benchmark:

```bash
python -m experiments.MEMS_end_of_line.run
```

Enable online experiment tracking (after `wandb login`):

```bash
python -m experiments.MEMS_end_of_line.run --wandb-mode online \
  --wandb-project MEMS_end_of_line --wandb-group mems-simulated-lstm \
  --epochs 10000 --patience 10000 --lr-scheduler
```

The Slurm GPU submission script uses online W&B tracking by default:

```bash
sbatch experiments/MEMS_end_of_line/slurm_train_gpu.sh
```

For a short smoke run:

```bash
python -m experiments.MEMS_end_of_line.run --epochs 2 --patience 2 --width 16 \
  --posterior-samples 4 --diffusion-steps 10 --flow-steps 10
```

Outputs in `results/MEMS_end_of_line` include one checkpoint and training curve per
model, posterior samples, `metrics.csv`, `model_comparison.png`, and one
`<model>_parameter_regression.png` figure per method. Each regression figure has
six panels showing posterior means and 90% credible intervals against the six
ground-truth MEMS parameters. Lower RMSE, MAE, and energy score are better;
coverage should be near 0.90.

Figures can also be regenerated from existing posterior samples:

```bash
python -m experiments.MEMS_end_of_line.plot_regressions \
  --results-dir results/MEMS_end_of_line
```
