#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_ddpm_curve
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=auxme-imech091
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_state_vs_ddpm_curve_evaluation/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_state_vs_ddpm_curve_evaluation/slurm-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
OUTPUT="$ROOT/results/results_optimal_control/door/csi_state_vs_ddpm_curve_evaluation"
mkdir -p "$OUTPUT"
cd "$ROOT"

export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON=/home/zhangyi/myenv_312/bin/python

nvidia-smi
"$PYTHON" - <<'PY'
import jax
print("jax", jax.__version__, "devices", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not discover the requested L40S GPU")
PY

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
  --manifest experiments/cond_stoch_interpolant_dppo/config/evaluation_door_csi_state_vs_ddpm_curve.json
