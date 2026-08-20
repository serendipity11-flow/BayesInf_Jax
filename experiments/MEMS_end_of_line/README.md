# MEMS end-of-line parameter inference

This experiment learns a conditional posterior over six MEMS parameters from
signals shaped `(batch, 96, 4)`. The deterministic seeded split contains 200
training, 50 validation, and 50 test examples. The encoder comparison uses the
source signals and parameters directly (`--no-signal-standardization`). The
data loader still supports training-set signal standardization for other runs;
parameters are never standardized by this experiment.

## Flow architecture and API

All three comparison models use four two-sided affine coupling blocks separated
by random permutations. Each block updates both three-dimensional halves of the
parameter vector. The coupling networks have four 64-unit hidden layers with
`tanh` activation and Xavier-uniform initialization.

The BatchNorm-free models use the general conditional-flow API:

```python
CondNormFlow(flow_type="double_realnvp", encoder_type="lstm")
CondNormFlow(flow_type="double_realnvp", encoder_type="transformer")
```

The stateful BatchNorm model is the sibling `BNNormFlow`. This separation keeps
BatchNorm running means and variances out of the stateless `CondNormFlow`
interface. The old notebook-specific `MEMSDoubleRealNVP` class is no longer
used.

The experiment names are:

- `lstm_bn_flow` uses a two-layer bidirectional LSTM, with a single time scan
  whose cell operations are vectorized over the full batch. Its two summary
  BatchNorm layers keep explicit running statistics, saved in
  `lstm_bn_flow.state.eqx`.
- `lstm_flow` uses the same vectorized recurrent layout without BatchNorm. Its
  projection activation is `tanh`.
- `transformer_flow` replaces that conditioner with a batch-first Transformer
  encoder using positional embeddings, multi-head self-attention, LayerNorm,
  and mean pooling. It contains no BatchNorm.

All models produce a 64-dimensional condition vector and use the same
double-sided RealNVP blocks and the notebook's
five data-initialized ActNorm transforms. Sampling keeps the complete
`(posterior samples, observations, parameters)` tensor vectorized through the
flow rather than flattening and repeating conditions.

## Raw-signal encoder feature comparison

`slurm_train_encoder_scheduler_gpu.sh` runs the matched three-way comparison on
the original signal and parameter values:

- `lstm_bn_flow`: vectorized bidirectional LSTM with BatchNorm;
- `lstm_flow`: the same vectorized bidirectional LSTM projections without
  BatchNorm and with `tanh` activation;
- `transformer_flow`: Transformer encoder with LayerNorm and no BatchNorm.

ActNorm is initialized from the complete training set and frozen: its stored
bias and log-scale are stop-gradient values and remain unchanged during
optimizer updates. After training, the job evaluates each posterior and runs
`plot_encoder_pca.py`. The resulting `encoder_pca_feature_maps.png` has one row
per encoder and one column per inferred parameter. The underlying 64-dimensional
features, two-dimensional PCA coordinates, split labels, and explained variance
are also saved in `encoder_pca_features.npz` and
`encoder_pca_explained_variance.csv`.

All three MEMS flows hold the learning rate at `1e-4` for 500 epochs and then
follow cosine decay to `1e-6`. During training, 10% dropout is applied to the
features conditioning the invertible flow; inference remains deterministic.

Training runs for up to 20,000 epochs. Early stopping uses validation NLL with
patience 1500 and restores the checkpoint with the best validation NLL.
The former `BayesFlow_MEMS_JAX_Equinox.py` monolith is now only a compatibility
entry point. Canonical networks, flows, data handling, training, evaluation,
and plotting live in their corresponding structured modules.

## Training

```bash
python -m experiments.MEMS_end_of_line.train_inference \
  --models lstm_bn_flow lstm_flow transformer_flow \
  --epochs 20000 --patience 1500 --batch-size 200 \
  --learning-rate 1e-4 --lr-hold-epochs 500 \
  --min-learning-rate-ratio 0.01 --flow-dropout 0.1 \
  --width 64 --encoder-layers 2 --transformer-heads 4 \
  --no-signal-standardization
```

Submit GPU training with:

```bash
mkdir -p results/results_parameter_identification/MEMS_end_of_line/encoder_flow_api_comparison/logs
sbatch experiments/MEMS_end_of_line/slurm_train_encoder_scheduler_gpu.sh
```

## Evaluation

Evaluation restores a checkpoint, draws posterior samples for the test split,
and writes metrics, samples, and a six-parameter regression figure.

```bash
python -m experiments.MEMS_end_of_line.evaluate_inference \
  --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_flow_api_comparison \
  --models lstm_bn_flow lstm_flow transformer_flow --posterior-samples 500
```

Generate the encoder PCA feature map:

```bash
python -m experiments.MEMS_end_of_line.plot_encoder_pca \
  --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_flow_api_comparison
```

Smoke test:

```bash
python -m experiments.MEMS_end_of_line.train_inference \
  --models lstm_bn_flow lstm_flow transformer_flow \
  --epochs 2 --patience 2 --batch-size 200 --width 8 \
  --encoder-layers 1 --transformer-heads 2 --no-signal-standardization
```
