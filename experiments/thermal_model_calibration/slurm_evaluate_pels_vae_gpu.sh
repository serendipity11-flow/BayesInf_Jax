#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=thermal_pels_eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/thermal_model_calibration/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/thermal_model_calibration/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR=/home/zhangyi/xrg-mixed-cabin-calibration-BayesFlow/simulation_data
CHECKPOINT_DIR="$PROJECT_ROOT/results/results_parameter_identification/thermal_model_calibration/pels_vae_normalized"
OUTPUT_DIR="$PROJECT_ROOT/results/results_parameter_identification/thermal_model_calibration/pels_vae_normalized_evaluation"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

/usr/bin/python3 -m experiments.thermal_model_calibration.evaluate_pels_vae \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --passes 20 \
    --batch-size 10 \
    --plot-indices 0 50 100 150 \
    --seed 0
