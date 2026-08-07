#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=offroad_v2_s50_energy
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=l40s-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/offroad_flowsde/logs/train_v2_seg50_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/offroad_flowsde/logs/train_v2_seg50_%j.err

set -euo pipefail
PROJECT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
DATA="$PROJECT/data/offroad_flowsde/offroad_v2.h5"
OUTPUT="$PROJECT/results/results_optimal_control/offroad_flowsde/v2_seg50_stride5_tw1_energy0p1_seed0"
DEPS="$PROJECT/data/offroad_flowsde/.slurm_python_packages"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT:$DEPS:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$PROJECT/results/results_optimal_control/offroad_flowsde/logs" "$MPLCONFIGDIR"
cd "$PROJECT"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3.12 || command -v python3)"; fi
"$PYTHON" -c 'import jax, equinox, optax, h5py; devices=jax.devices(); print("JAX devices:",devices); assert any(d.platform=="gpu" for d in devices)'
"$PYTHON" -m experiments.offroad_flowsde.train_segments \
  --data "$DATA" --output "$OUTPUT" \
  --seg-len 50 --stride 5 --transition-weight 1.0 --energy-weight 0.1 \
  --particles 4 --epochs 100 --batch-size 128 --validation-segments 1024 \
  --samples-per-epoch 20000 \
  --lr 1e-4 --grad-clip 1.0 --dt 0.01 \
  --width 256 --depth 2 --condition-dim 32 --flow-layers 6 --seed 0
