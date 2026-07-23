#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=lane_flow_mpc
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%5
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/%x_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$MPLCONFIGDIR"

VENV_SITE="$HOME/myenv_312/lib/python3.12/site-packages"
export PYTHONPATH="$PROJECT_ROOT:$VENV_SITE:${PYTHONPATH:-}"
if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, optax, scipy' >/dev/null 2>&1; then
    PYTHON="$HOME/myenv_312/bin/python"
else
    PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "$PYTHON" ]]; then
    echo "Could not locate Python 3.12" >&2
    exit 1
fi

SEED="${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/runs/seed${SEED}"
CHECKPOINT="$OUTPUT_DIR/lane_flow.eqx"
CSV="$OUTPUT_DIR/comparison.csv"
mkdir -p "$OUTPUT_DIR"

echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "seed=$SEED output=$OUTPUT_DIR"

"$PYTHON" -m bayesinf.applications.optimal_control.train_lane_keeping \
    --samples 12000 \
    --epochs 200 \
    --batch-size 128 \
    --learning-rate 3e-4 \
    --seed "$SEED" \
    --output "$CHECKPOINT"

"$PYTHON" -m bayesinf.applications.optimal_control.evaluate_lane_keeping \
    --checkpoint "$CHECKPOINT" \
    --seed "$SEED" \
    --steps 300 \
    --csv "$CSV"

echo "Completed seed=$SEED checkpoint=$CHECKPOINT comparison=$CSV"
