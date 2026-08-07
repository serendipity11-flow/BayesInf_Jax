#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_path_bc
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-path-bc-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-path-bc-%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/door-expert-v2-100"
SEED="$SLURM_ARRAY_TASK_ID"
OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/door/si_analytic_path_v5/seed$SEED/pretrain"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/results/results_optimal_control/slurm"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

PYTHON=""
for candidate in "$HOME/myenv_312/bin/python" /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "No working Python 3.12 interpreter found on $(hostname)" >&2
    exit 127
fi

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
    --policy interpolant --data "$DATA_DIR" --output "$OUTPUT_DIR" \
    --env AdroitHandDoor-v2 --max-episode-steps 300 \
    --obs-dim 39 --action-dim 28 --horizon 4 --cond-steps 1 \
    --seed "$SEED" --gradient-steps 100000 --batch-size 128 \
    --learning-rate 1e-3 --weight-decay 1e-6 --log-every 1000 \
    --integration-steps 50 --gaussian-base --sigma-0 0.1 \
    --sampling-std 0.1 --score-sigma-floor 0.02 --sde-eta 0.05 \
    --path-loss-weight 0.1 --path-loss-steps 10 \
    --hidden-size 52 --depth 2 --time-dim 16 --context-dim 32 --n-heads 4
