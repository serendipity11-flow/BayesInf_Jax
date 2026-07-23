#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_dppo_jax
#SBATCH --partition=a30
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-gpu
#SBATCH --output=/home/zhangyi/bayesinf_jax/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/dppo/door-human-v2"
OUTPUT_DIR="$PROJECT_ROOT/results/cond_stoch_interpolant_dppo/door/dppo_jax/seed0/pretrain"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
if [[ -x "$HOME/myenv_312/bin/python" ]]; then
    PYTHON="$HOME/myenv_312/bin/python"
else
    PYTHON="$(command -v python3.12)"
fi

echo "job_id=$SLURM_JOB_ID host=$(hostname) python=$PYTHON"
nvidia-smi || true
"$PYTHON" - <<'PY'
import jax
print("jax", jax.__version__, "devices", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not discover the requested GPU")
PY

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.validate_data \
    "$DATA_DIR" --obs-dim 39 --action-dim 28 --horizon 4

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
    --policy dppo \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --obs-dim 39 \
    --action-dim 28 \
    --horizon 4 \
    --cond-steps 1 \
    --denoising-steps 20 \
    --time-dim 16 \
    --sampling-std 0.1 \
    --gradient-steps 100000 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --weight-decay 1e-6 \
    --log-every 1000 \
    --seed 0
