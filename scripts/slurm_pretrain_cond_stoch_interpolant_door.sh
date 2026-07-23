#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:a30:[1-4]":
#SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-gpu

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/dppo/door-human-v2"
OUTPUT_DIR="${COND_SI_OUTPUT_DIR:-$PROJECT_ROOT/results/cond_stoch_interpolant_dppo/door/si_informative/seed0/pretrain}"
TRAIN_STEPS="${COND_SI_TRAIN_STEPS:-100000}"
LOG_EVERY="${COND_SI_LOG_EVERY:-1000}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/results/cond_stoch_interpolant_dppo/slurm"

export PYTHONNOUSERSITE=1
VENV_SITE="$HOME/myenv_312/lib/python3.12/site-packages"
export PYTHONPATH="$PROJECT_ROOT:$VENV_SITE:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

if [[ -x "$HOME/myenv_312/bin/python" ]]; then
    PYTHON="$HOME/myenv_312/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.12)"
elif [[ -x /usr/local/bin/python3.12 ]]; then
    PYTHON=/usr/local/bin/python3.12
else
    echo "No Python 3.12 interpreter found" >&2
    exit 127
fi

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "python=$PYTHON ($("$PYTHON" --version))"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "data=$DATA_DIR"
echo "output=$OUTPUT_DIR"
echo "train_steps=$TRAIN_STEPS"
nvidia-smi || true

"$PYTHON" - <<'PY'
import jax
import equinox
import optax
print("jax", jax.__version__)
print("equinox", equinox.__version__)
print("optax", optax.__version__)
print("devices", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("Slurm job requested a GPU but JAX did not discover one")
PY

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.validate_data \
    "$DATA_DIR" \
    --obs-dim 39 \
    --action-dim 28 \
    --horizon 4

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --obs-dim 39 \
    --action-dim 28 \
    --horizon 4 \
    --cond-steps 1 \
    --integration-steps 5 \
    --sigma-0 0.1 \
    --sampling-std 0.1 \
    --hidden-size 64 \
    --depth 2 \
    --time-dim 16 \
    --context-dim 32 \
    --n-heads 4 \
    --gradient-steps "$TRAIN_STEPS" \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --weight-decay 1e-6 \
    --log-every "$LOG_EVERY" \
    --seed 0
