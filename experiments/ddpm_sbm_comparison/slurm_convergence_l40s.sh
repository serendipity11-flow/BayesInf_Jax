#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=ddpm_sbm_limit
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON="$HOME/myenv_312/bin/python"
else
    PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "$PYTHON" ]]; then
    PYTHON="$(find /usr/bin /usr/local/bin -maxdepth 1 -name 'python3.12' -type f 2>/dev/null | head -1)"
fi
if [[ -z "$PYTHON" ]]; then
    echo "Could not locate Python 3.12" >&2
    exit 1
fi

CASES=(basic linear multimodal)
CASE="${CASES[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 5))
INPUT="$PROJECT_ROOT/experiments/ddpm_sbm_comparison/results/${CASE}_seed${SEED}"

echo "Evaluating same-network convergence: case=$CASE seed=$SEED"
"$PYTHON" -m experiments.ddpm_sbm_comparison.evaluate_convergence \
    --input "$INPUT" \
    --steps 25 50 100 250 500 1000 \
    --posterior-samples 1000
