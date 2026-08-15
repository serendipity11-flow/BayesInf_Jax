#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_bf_ref_jax
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/MEMS_end_of_line/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/MEMS_end_of_line/logs/%x_%j.err

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON=/usr/bin/python3

echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" - <<'PY'
import wandb
if not wandb.api.api_key:
    raise RuntimeError("W&B login is not configured")
print("W&B authentication is configured")
PY

"$PYTHON" -m experiments.MEMS_end_of_line.run_bayesflow_reference \
  --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
  --output-dir results/results_parameter_identification/MEMS_end_of_line/bayesflow_reference_jax \
  --epochs 20000 \
  --posterior-samples 100 \
  --seed 0 \
  --wandb-mode online \
  --wandb-project MEMS_end_of_line \
  --wandb-group bayesflow-reference-jax
