# Thermal model calibration

This experiment ports the iterative workflow from
`xrg-mixed-cabin-calibration-BayesFlow` to JAX. The sensitivity-derived order is
`[11,14,15,16]`, `[12,13,17]`, `[6,7,8,10,18]`, `[0,1,2,3,4]`, `[5,9]`.
The first inverse model is trained on FMU-generated trajectories. After each
posterior estimate, identified parameters are fixed and the pretrained PELS-VAE
generates the training and validation trajectories for the next group. A noisy
proxy-real observation remains the calibration target at every stage.

Run from the repository root:

```bash
python -m experiments.thermal_model_calibration.run \
  --data-dir simulation_data \
  --vae-checkpoint results/checkpoints/pels_vae.eqx
```

The input directory must contain `data_T_HeatUp_CoolDown_20%.npy` and
`data_parameter_HeatUp_CoolDown_20%.npy`. The final 50 trajectories are reserved
as a proxy real-test set and receive reproducible Gaussian observation noise
(standard deviation 0.05 degrees). The remaining trajectories are split
80/10/10 into training, validation, and clean test sets. Validation supports
model selection during training, the clean test set evaluates trained models,
and only the noisy real-test proxy drives the complete iterative calibration.
All normalization statistics are learned from training data only. Results and
one checkpoint/history/posterior bundle per stage are written by default to
`results/results_parameter_identification/thermal_model_calibration`.

The legacy `.pt` checkpoint contains PyTorch/FrEIA tensors and is not directly
loadable by Equinox. `--vae-checkpoint` therefore expects a pretrained `.eqx`
PELS-VAE with the architecture arguments matching those passed to this runner.

Train that surrogate with the translated reconstruction, latent-matching, and
scheduled-KL objective using:

```bash
python -m experiments.thermal_model_calibration.train_pels_vae \
  --data-dir simulation_data
```

This saves both `pels_vae.eqx` and the required `pels_vae_state.eqx`. The inverse
stages use the adjusted MEMS `BNNormFlow`: four two-sided coupling blocks with
four 64-unit tanh layers per subnet, ActNorm between blocks, and the stateful
BatchNorm LSTM conditioner. Each stage saves its BatchNorm state separately and records surrogate-generation,
preprocessing, training, posterior-evaluation, and total wall-clock durations in
`timing.json`; the complete workflow writes `timings.csv` and
`timing_summary.json`.

Submit the complete iterative workflow on L40S with:

```bash
sbatch experiments/thermal_model_calibration/slurm_iterative_double_realnvp_gpu.sh
```
