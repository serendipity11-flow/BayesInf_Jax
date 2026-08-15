#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_ft
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-ft-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-ft-%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
DATA_DIR="$PROJECT_ROOT/data/door-expert-v2-100"
SEEDS=3
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))

case "$METHOD_INDEX" in
    0)
        POLICY=csi_sde
        METHOD_DIR=csi_sde_v2
        CHECKPOINT="$PROJECT_ROOT/results/results_optimal_control/door/csi_sde_v2/seed$SEED/pretrain/checkpoint"
        POLICY_ARGS=(--integration-steps 20
                     --sigma-0 0.1 --sampling-std 0.1
                     --score-sigma-floor 0.02 --sde-eta 0.005
                     --path-loss-weight 0.2 --path-loss-steps 20
                     --score-head-width 16 --score-loss-weight 0.05
                     --score-correction-scale 0.1
                     --hidden-size 52 --depth 3
                     --time-dim 16 --context-dim 64 --n-heads 4)
        ;;
    1)
        POLICY=dppo
        METHOD_DIR=diffusion_dppo_v2
        CHECKPOINT="$PROJECT_ROOT/results/results_optimal_control/door/diffusion_dppo_v2/seed$SEED/pretrain/checkpoint"
        POLICY_ARGS=(--denoising-steps 20 --time-dim 16)
        ;;
    *)
        echo "Unexpected array index: $SLURM_ARRAY_TASK_ID" >&2
        exit 2
        ;;
esac

OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/door/$METHOD_DIR/seed$SEED/finetune"
if [[ ! -f "$CHECKPOINT/policy.eqx" ]]; then
    echo "Missing pretraining checkpoint: $CHECKPOINT/policy.eqx" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/results/results_optimal_control/slurm"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MUJOCO_GL=egl
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

echo "job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID policy=$POLICY"
echo "host=$(hostname) python=$PYTHON checkpoint=$CHECKPOINT output=$OUTPUT_DIR"
nvidia-smi || true

"$PYTHON" - <<'PY'
import jax
print("jax", jax.__version__, "devices", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not discover the requested GPU")
PY

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax finetune \
    --policy "$POLICY" \
    --data "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --env AdroitHandDoor-v2 \
    --max-episode-steps 300 \
    --obs-dim 39 \
    --action-dim 28 \
    --horizon 4 \
    --execution-horizon 4 \
    --cond-steps 1 \
    --seed "$SEED" \
    --iterations 75 \
    --checkpoint-every 5 \
    --n-envs 40 \
    --rollout-steps 70 \
    --update-epochs 2 \
    --actor-learning-rate 1e-5 \
    --critic-learning-rate 1e-3 \
    --critic-hidden-size 256 \
    --critic-depth 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --gamma-denoising 0.99 \
    --clip 0.01 \
    --clip-base 0.01 \
    --clip-rate 3 \
    --vf-coef 0.5 \
    --target-kl 5e-5 \
    --bc-reference-coef 0.05 \
    --reward-scale 0.01 \
    --max-grad-norm 1.0 \
    "${POLICY_ARGS[@]}"
