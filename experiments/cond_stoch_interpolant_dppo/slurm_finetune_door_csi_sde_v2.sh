#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_sde_v2_ft
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-sde-v2-ft-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-sde-v2-ft-%A_%a.out
set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax; SEED="$SLURM_ARRAY_TASK_ID"; DATA="$ROOT/data/door-expert-v2-100"
CKPT="$ROOT/results/results_optimal_control/door/csi_sde_v2/seed$SEED/pretrain/checkpoint"
OUT="$ROOT/results/results_optimal_control/door/csi_sde_v2/seed$SEED/finetune"
cd "$ROOT"; mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_ARRAY_JOB_ID}-${SEED}"
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
 --policy csi_sde --data "$DATA" --checkpoint "$CKPT" --output "$OUT" \
 --env AdroitHandDoor-v2 --max-episode-steps 300 --obs-dim 39 --action-dim 28 \
 --horizon 4 --execution-horizon 4 --cond-steps 1 --seed "$SEED" \
 --integration-steps 20 --sigma-0 .1 --sde-eta .005 --score-sigma-floor .02 \
 --path-loss-weight .2 --path-loss-steps 20 --score-head-width 16 \
 --score-loss-weight .05 --score-correction-scale .1 --hidden-size 52 --depth 3 \
 --context-dim 64 --time-dim 16 --n-heads 4 --iterations 75 --checkpoint-every 10 \
 --n-envs 40 --rollout-steps 70 --update-epochs 10 --actor-learning-rate 1e-4 \
 --critic-learning-rate 1e-3 --gamma .99 --gae-lambda .95 --gamma-denoising .99 \
 --clip .01 --clip-base .01 --clip-rate 3 --vf-coef .5 --target-kl 1
