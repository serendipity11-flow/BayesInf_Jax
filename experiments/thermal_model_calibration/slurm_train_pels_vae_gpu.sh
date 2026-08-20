#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=thermal_pels_vae
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
OUTPUT_DIR="$PROJECT_ROOT/results/results_parameter_identification/thermal_model_calibration/pels_vae_normalized"
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
devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("The PELS-VAE training job requires a JAX GPU device")
PY

"$PYTHON" -m experiments.thermal_model_calibration.train_pels_vae \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10000 \
    --patience 1000 \
    --batch-size 100 \
    --learning-rate 1e-3 \
    --latent-dim 64 \
    --hidden-dim 512 \
    --seed 0
