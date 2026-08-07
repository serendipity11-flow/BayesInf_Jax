#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_ft_safe
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-ft-safe-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-ft-safe-%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/door-expert-v2-100"
SEEDS=3
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))

if [[ "$METHOD_INDEX" -eq 0 ]]; then
    POLICY=interpolant
    METHOD_DIR=si_conservative_v6
    CHECKPOINT="$PROJECT_ROOT/results/results_optimal_control/door/si_analytic_path_v5/seed$SEED/pretrain/checkpoint"
    POLICY_ARGS=(--integration-steps 50 --gaussian-base
                 --sigma-0 0.1 --sampling-std 0.1
                 --score-sigma-floor 0.02 --sde-eta 0.01
                 --path-loss-weight 0.1 --path-loss-steps 10
                 --hidden-size 52 --depth 2 --time-dim 16
                 --context-dim 32 --n-heads 4)
else
    POLICY=dppo
    METHOD_DIR=diffusion_conservative_v6
    CHECKPOINT="$PROJECT_ROOT/results/results_optimal_control/door/diffusion_dppo_v2/seed$SEED/pretrain/checkpoint"
    POLICY_ARGS=(--denoising-steps 20 --ft-denoising-steps 10
                 --sampling-std 0.1 --time-dim 16)
fi

OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/door/$METHOD_DIR/seed$SEED/finetune"
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

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax finetune \
    --policy "$POLICY" --data "$DATA_DIR" --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR" --env AdroitHandDoor-v2 --max-episode-steps 300 \
    --obs-dim 39 --action-dim 28 --horizon 4 --execution-horizon 4 \
    --cond-steps 1 --seed "$SEED" --iterations 30 --checkpoint-every 5 \
    --n-envs 40 --rollout-steps 70 --update-epochs 2 \
    --actor-learning-rate 1e-5 --critic-learning-rate 1e-3 \
    --gamma 0.99 --gae-lambda 0.95 --gamma-denoising 0.99 \
    --clip 0.01 --clip-base 0.01 --clip-rate 3 \
    --vf-coef 0.5 --target-kl 0.005 --bc-reference-coef 0.05 \
    --reward-scale 0.01 --max-grad-norm 1.0 \
    "${POLICY_ARGS[@]}"
