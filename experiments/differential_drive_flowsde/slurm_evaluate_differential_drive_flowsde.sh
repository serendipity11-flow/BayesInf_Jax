#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flowsde_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/logs/evaluate_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/logs/evaluate_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
DATA=${DATA:-"$PROJECT_ROOT/data/differential_drive/data.h5"}
RESULT=${RESULT:-"$PROJECT_ROOT/results/results_optimal_control/differential_drive/seed0/joint_stage"}
OUTPUT=${OUTPUT:-"$PROJECT_ROOT/results/results_optimal_control/differential_drive/seed0/evaluation"}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-eval-${SLURM_JOB_ID}"
export JAX_PLATFORMS=cpu

mkdir -p "$MPLCONFIGDIR" "$OUTPUT"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate \
    --data "$DATA" \
    --result "$RESULT" \
    --output "$OUTPUT" \
    --particles 100 \
    --calibration-transitions 2000 \
    --rollout-episodes 10 \
    --horizon 100 \
    --seed 17
