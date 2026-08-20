#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=thermal_iterative_bf
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/thermal_model_calibration/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/thermal_model_calibration/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR=/home/zhangyi/xrg-mixed-cabin-calibration-BayesFlow/simulation_data
PELS_DIR="$PROJECT_ROOT/results/results_parameter_identification/thermal_model_calibration/pels_vae_normalized"
OUTPUT_DIR="$PROJECT_ROOT/results/results_parameter_identification/thermal_model_calibration/iterative_parameter_identification"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

PYTHON=/usr/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" - <<'PY'
import jax
import wandb
devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("Iterative BayesFlow training requires a JAX GPU device")
if not wandb.api.api_key:
    raise RuntimeError("W&B is not authenticated")
print("W&B authentication is configured")
PY

"$PYTHON" -m experiments.thermal_model_calibration.run \
    --data-dir "$DATA_DIR" \
    --vae-checkpoint "$PELS_DIR/pels_vae.eqx" \
    --vae-state "$PELS_DIR/pels_vae_state.eqx" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10000 \
    --patience 1000 \
    --batch-size 200 \
    --learning-rate 5e-4 \
    --min-learning-rate 5e-6 \
    --lr-hold-epochs 500 \
    --posterior-samples 100 \
    --width 64 \
    --rnn-hidden 64 \
    --encoder-layers 2 \
    --latent-dim 64 \
    --vae-hidden 512 \
    --vae-passes 1 \
    --intermediate-eval-every 100 \
    --intermediate-posterior-samples 100 \
    --wandb-mode online \
    --wandb-project thermal_model_calibration \
    --wandb-group iterative-double-realnvp \
    --real-test-index 0 \
    --seed 0
