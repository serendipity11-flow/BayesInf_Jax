#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_bc
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-bc-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-bc-%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/door-expert-v2-100"
SEEDS=3
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))

if [[ "$METHOD_INDEX" -eq 0 ]]; then
    POLICY=interpolant
    METHOD_DIR=si_analytic_dppo_v4
    POLICY_ARGS=(--integration-steps 50 --gaussian-base
                 --sigma-0 0.1 --sampling-std 0.1
                 --score-sigma-floor 0.02 --sde-eta 0.05
                 --hidden-size 52 --depth 2 --time-dim 16
                 --context-dim 32 --n-heads 4)
else
    POLICY=dppo
    METHOD_DIR=diffusion_dppo_v2
    POLICY_ARGS=(--denoising-steps 20 --sampling-std 0.1 --time-dim 16)
fi

OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/door/$METHOD_DIR/seed$SEED/pretrain"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/results/results_optimal_control/slurm"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON=""
for candidate in \
    "$HOME/myenv_312/bin/python" \
    /usr/bin/python3.12 \
    /usr/local/bin/python3.12
do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "No working Python 3.12 interpreter found on $(hostname)" >&2
    exit 127
fi

echo "job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID policy=$POLICY seed=$SEED"
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.check_door_compatibility

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
    --policy "$POLICY" \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --env AdroitHandDoor-v2 \
    --max-episode-steps 300 \
    --obs-dim 39 \
    --action-dim 28 \
    --horizon 4 \
    --cond-steps 1 \
    --seed "$SEED" \
    --gradient-steps 100000 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --weight-decay 1e-6 \
    --log-every 1000 \
    "${POLICY_ARGS[@]}"
