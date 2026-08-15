#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_state_ft
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=auxme-imech091
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-ft-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-ft-%A_%a.out

set -euo pipefail

ROOT=/home/zhangyi/bayesinf_jax
SEED=${SLURM_ARRAY_TASK_ID}
CHECKPOINT="$ROOT/results/results_optimal_control/door/csi_state_equivalent_v1_20step/seed$SEED/pretrain/checkpoint"
OUTPUT="$ROOT/results/results_optimal_control/door/csi_state_equivalent_v1_20step/seed$SEED/finetune"
test -f "$CHECKPOINT/policy.eqx"
mkdir -p "$OUTPUT" "$ROOT/results/results_optimal_control/slurm"
cd "$ROOT"

PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  for candidate in /home/zhangyi/myenv_312/bin/python /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import jax, equinox, gymnasium_robotics' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "No compatible Door/JAX Python found on $(hostname)" >&2
  exit 127
fi

export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "host=$(hostname) python=$PYTHON policy=csi_state_equivalent seed=$SEED"
nvidia-smi
"$PYTHON" - <<'PY'
import jax
print("jax", jax.__version__, "devices", jax.devices())
if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not discover the requested L40S GPU")
PY

# Every environment, rollout, optimizer, PPO, and checkpoint setting below is
# copied from the completed diffusion_dppo_v2 Door fine-tuning configuration.
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax finetune \
  --policy interpolant --score-mode learned \
  --data data/door-expert-v2-100 \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --env AdroitHandDoor-v2 --max-episode-steps 300 \
  --obs-dim 39 --action-dim 28 --horizon 4 --execution-horizon 4 --cond-steps 1 \
  --encoder-type mlp --integration-steps 20 --ft-denoising-steps 10 \
  --sigma-0 1.0 --score-sigma-floor 0.05 \
  --sde-eta 0.25 --sampling-std 0.001 \
  --score-loss-weight 0.25 \
  --hidden-size 128 --score-head-width 128 --depth 3 \
  --context-dim 64 --time-dim 32 --n-heads 4 \
  --seed "$SEED" --iterations 75 --checkpoint-every 5 \
  --n-envs 40 --rollout-steps 70 --update-epochs 2 \
  --actor-learning-rate 1e-5 --critic-learning-rate 1e-3 \
  --critic-hidden-size 256 --critic-depth 3 --max-grad-norm 1.0 \
  --gamma 0.99 --gae-lambda 0.95 --gamma-denoising 0.99 \
  --clip 0.01 --clip-base 0.01 --clip-rate 3 \
  --vf-coef 0.5 --target-kl 5e-5 \
  --bc-reference-coef 0.05 --reward-scale 0.01
