#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_data_extended_postfix
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/data/differential_drive/logs/extended_state_postfix-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/data/differential_drive/logs/extended_state_postfix-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
OUTPUT="$PROJECT_ROOT/data/differential_drive/smooth_terrain_navigation_extended_state_postfix.h5"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$PROJECT_ROOT/data/differential_drive/logs"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m experiments.differential_drive_flowsde.dataset \
  --output "$OUTPUT" --trajectories 300 --split-trajectories 210 45 45 \
  --steps 500 --waypoint-steps 125 --command-smoothing .08 \
  --seed 2026 --extended-state
