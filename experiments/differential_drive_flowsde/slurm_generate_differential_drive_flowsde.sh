#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flowsde_data
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/data/differential_drive/logs/smooth_terrain_generate_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/data/differential_drive/logs/smooth_terrain_generate_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
OUTPUT="$PROJECT_ROOT/data/differential_drive/smooth_terrain_navigation.h5"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.dataset \
    --output "$OUTPUT" \
    --trajectories 300 \
    --split-trajectories 210 45 45 \
    --steps 500 \
    --waypoint-steps 125 \
    --command-smoothing 0.08 \
    --dt 0.02 \
    --seed 2026

"$PYTHON_BIN" -c "import h5py; f=h5py.File('$OUTPUT'); print({k: v.shape for k, v in f.items()}); print(dict(f.attrs)); f.close()"
