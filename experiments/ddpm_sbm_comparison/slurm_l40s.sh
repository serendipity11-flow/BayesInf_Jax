#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=ddpm_sbm_msd
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-14%5
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/ddpm_sbm_comparison/logs/%x_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/ddpm_sbm_comparison/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

VENV_SITE="$HOME/myenv_312/lib/python3.12/site-packages"
export PYTHONPATH="$PROJECT_ROOT:$VENV_SITE:${PYTHONPATH:-}"
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON="$HOME/myenv_312/bin/python"
else
    PYTHON="$(command -v python3.12)"
fi

CASES=(basic linear multimodal)
SEEDS=(0 1 2 3 4)
CASE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
CASE="${CASES[$CASE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="$PROJECT_ROOT/experiments/ddpm_sbm_comparison/results/${CASE}_seed${SEED}"

echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case=$CASE seed=$SEED output=$OUTPUT_DIR"

"$PYTHON" -m experiments.ddpm_sbm_comparison.run \
    --cases "$CASE" \
    --seeds "$SEED" \
    --train-size 4096 \
    --validation-size 512 \
    --test-size 200 \
    --epochs 2000 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --patience 200 \
    --width 128 \
    --depth 3 \
    --dt 0.05 \
    --duration 1.0 \
    --noise-std 0.0 \
    --sample-steps 25 50 100 250 500 1000 \
    --posterior-samples 1000 \
    --output "$OUTPUT_DIR"
