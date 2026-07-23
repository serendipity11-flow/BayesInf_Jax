#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_fsde_s16_u08
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de

set -euo pipefail

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:${PYTHONPATH:-}

if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, optax, pandas, pyarrow' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || true)
fi
if [ -z "$PYTHON" ]; then
    echo "Python 3.12 environment not found" >&2
    exit 1
fi

cd /home/zhangyi/bayesinf_jax
UNCERTAINTY_WEIGHT=${UNCERTAINTY_WEIGHT:-0.6}
CONTINUITY_WEIGHT=${CONTINUITY_WEIGHT:-0.1}
SEGMENT_LENGTH=${SEGMENT_LENGTH:-16}
EPOCHS=${EPOCHS:-50}
DATA_PATH=${DATA_PATH:-data/aloha_mobile_cabinet/file-000.parquet}
TIME_STEP=${TIME_STEP:-0.02}
WINDOW_STEPS=${WINDOW_STEPS:-128}
WINDOW_STRIDE=${WINDOW_STRIDE:-16}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-2048}
OUTPUT=${FLOW_OUTPUT:-results/results_optimal_control/aloha_mobile_cabinet/flow_sde_dense_uw0p8}
DIFFUSION_NO_CONTROL=${DIFFUSION_NO_CONTROL:-0}
VELOCITY_AUGMENTED=${VELOCITY_AUGMENTED:-0}
DIFFUSION_TAG=""
DIFFUSION_ARGS=()
if [ "$VELOCITY_AUGMENTED" = "1" ]; then
    DIFFUSION_TAG="_qv"
    DIFFUSION_ARGS+=(--velocity-augmented --velocity-window 3)
elif [ "$DIFFUSION_NO_CONTROL" = "1" ]; then
    DIFFUSION_TAG="_gx"
    DIFFUSION_ARGS+=(--diffusion-no-control)
fi
CONFIG=$OUTPUT/aloha_flow_sde_heun_seg${SEGMENT_LENGTH}${DIFFUSION_TAG}_noise0p01_seed0_config.json

echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import jax, pandas, pyarrow; print("JAX devices:", jax.devices()); print("pyarrow:", pyarrow.__version__)'
echo "Training dense-start ALOHA FlowSDE, epochs=$EPOCHS, segment=$SEGMENT_LENGTH, uncertainty_weight=$UNCERTAINTY_WEIGHT, continuity_weight=$CONTINUITY_WEIGHT, diffusion_no_control=$DIFFUSION_NO_CONTROL, velocity_augmented=$VELOCITY_AUGMENTED"

"$PYTHON" -m bayesinf.applications.optimal_control.train_aloha_flow_sde \
    --data "$DATA_PATH" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --dt "$TIME_STEP" \
    --batch-size 8 \
    --window-steps "$WINDOW_STEPS" \
    --segment-length "$SEGMENT_LENGTH" \
    --stride "$WINDOW_STRIDE" \
    --samples-per-epoch "$SAMPLES_PER_EPOCH" \
    --train-particles 4 \
    --solver heun \
    --uncertainty-weight "$UNCERTAINTY_WEIGHT" \
    --continuity-weight "$CONTINUITY_WEIGHT" \
    --state-noise-std 0.01 \
    --control-noise-std 0.01 \
    --validation-windows 16 \
    "${DIFFUSION_ARGS[@]}"

echo "Evaluating 64-particle full-horizon trajectories"
"$PYTHON" -m bayesinf.applications.optimal_control.eval_aloha_flow_sde \
    --data "$DATA_PATH" \
    --config "$CONFIG" \
    --output "$OUTPUT/evaluation" \
    --particles 64

echo "ALOHA FlowSDE training and evaluation complete"
