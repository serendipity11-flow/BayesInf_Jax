#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=lane_flow_recovery
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/%x_%j.err

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
    PYTHON="$(command -v python3.12)"
fi

OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/runs/seed0_recovery_nmpc"
mkdir -p "$OUTPUT_DIR"

"$PYTHON" -m bayesinf.applications.optimal_control.train_lane_keeping \
    --samples 12000 --epochs 200 --batch-size 128 --learning-rate 3e-4 \
    --first-action-weight 0 --rollout-weight 0 \
    --seed 0 --output "$OUTPUT_DIR/lane_flow.eqx"

"$PYTHON" -m bayesinf.applications.optimal_control.evaluate_lane_keeping \
    --checkpoint "$OUTPUT_DIR/lane_flow.eqx" --seed 0 --steps 300 \
    --csv "$OUTPUT_DIR/comparison.csv"
