#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=flow_sde
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=16G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:a30:[1-4]":
#SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-gpu

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"


# ---------------------------------------------------------------------------
# Fixed data / architecture settings
# ---------------------------------------------------------------------------
DATA_ARGS="
    --N 12 --obs_interval 2 --obs_std 0.5
    --horizon 101 --dt 0.02 --F_lorenz 8.0
    --spinup_steps 500 --n_samples 2000"

NOISE_ARGS="
    --obs_noise_type gmm
    --gmm_weights 0.70,0.15,0.15
    --gmm_offsets 0.0,2.5,-2.5
    --gmm_stds 0.50,1.50,1.50"

ARCH_ARGS="
    --d_c 128 --d_hidden_enc 128
    --d_hidden 256 --depth 3
    --d_hidden_diffusion 64
    --observation_stride 1
    --time_emb_dim 32"

N_REALNVP_LAYERS="${N_REALNVP_LAYERS:-2}"
FLOW_TYPE="${FLOW_TYPE:-realnvp}"
N_NICE_LAYERS="${N_NICE_LAYERS:-2}"
NICE_NOISE_SCALE="${NICE_NOISE_SCALE:-0.5}"

TRAIN_ARGS="
    --batch_size 32
    --learning_rate 3e-4 --grad_norm 1.0 --weight_decay 1e-4
    --patience 40 --max_epochs 500
    --n_realnvp_layers ${N_REALNVP_LAYERS} --realnvp_alpha 1.0
    --flow_type ${FLOW_TYPE} --n_nice_layers ${N_NICE_LAYERS}
    --nice_noise_scale ${NICE_NOISE_SCALE}
    --solver heun
    --use_custom_solver"

# Multiple-shooting setting. The best calibrated current setup uses provided
# segment starts, short segments, mild continuity regularisation, and a small
# amount of segment-start noise.
SEG_LIST="${SEG_LIST:-5}"
SEG_LENGTH_START="${SEG_LENGTH_START:-5}"
SEG_SCHEDULE_EPOCHS="${SEG_SCHEDULE_EPOCHS:-0}"
SHOOTING_NOISE_STD="${SHOOTING_NOISE_STD:-0.01}"
ROLLOUT_LOSS_WEIGHT="${ROLLOUT_LOSS_WEIGHT:-1}"
ROLLOUT_LOSS_TYPE="${ROLLOUT_LOSS_TYPE:-energy_score}"
ROLLOUT_NUM_ENSEMBLE="${ROLLOUT_NUM_ENSEMBLE:-5}"
ROLLOUT_ENERGY_BETA="${ROLLOUT_ENERGY_BETA:-1.0}"
ROLLOUT_UNCERTAINTY_WEIGHT="${ROLLOUT_UNCERTAINTY_WEIGHT:-0.6}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-5}"
ROLLOUT_WARMUP_EPOCHS="${ROLLOUT_WARMUP_EPOCHS:-0}"
ROLLOUT_WINDOW="${ROLLOUT_WINDOW:-100}"
CONTINUITY_LOSS_WEIGHT="${CONTINUITY_LOSS_WEIGHT:-0.1}"
INIT_LOSS_WEIGHT="${INIT_LOSS_WEIGHT:-1.0}"
# One initialization mode per Slurm job. Submit the comparison as:
#   sbatch --export=ALL,INITIAL_STATE_MODE=inferred scripts/slurm_flow_sde.sh
#   sbatch --export=ALL,INITIAL_STATE_MODE=provided scripts/slurm_flow_sde.sh
INITIAL_STATE_MODE="${INITIAL_STATE_MODE:-provided}"
if [[ "$INITIAL_STATE_MODE" != "inferred" && "$INITIAL_STATE_MODE" != "provided" ]]; then
    echo "INITIAL_STATE_MODE must be 'inferred' or 'provided'; got '$INITIAL_STATE_MODE'" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# RealNVP transition-NLL training with the explicit Heun solver
# ---------------------------------------------------------------------------
for SEG in $SEG_LIST; do
    echo "========================================"
    echo "  FlowSDE  flow=${FLOW_TYPE}  rnvp_layers=${N_REALNVP_LAYERS}  nice_layers=${N_NICE_LAYERS}  nice_scale=${NICE_NOISE_SCALE}"
    echo "  initial_state_mode=${INITIAL_STATE_MODE}"
    echo "  shooting_noise_std=${SHOOTING_NOISE_STD}  rollout_loss_weight=${ROLLOUT_LOSS_WEIGHT}"
    echo "  rollout_loss_type=${ROLLOUT_LOSS_TYPE}  rollout_num_ensemble=${ROLLOUT_NUM_ENSEMBLE}"
    echo "  rollout_energy_beta=${ROLLOUT_ENERGY_BETA}  rollout_uncertainty_weight=${ROLLOUT_UNCERTAINTY_WEIGHT}"
    echo "  rollout_every=${ROLLOUT_EVERY}  rollout_warmup_epochs=${ROLLOUT_WARMUP_EPOCHS}  rollout_window=${ROLLOUT_WINDOW}"
    echo "  continuity_loss_weight=${CONTINUITY_LOSS_WEIGHT}  init_loss_weight=${INIT_LOSS_WEIGHT}"
    echo "========================================"
    $PYTHON -m bayesinf.applications.state_estimation.train_flow_sde_lorenz96 \
        --train \
        $DATA_ARGS $NOISE_ARGS $ARCH_ARGS $TRAIN_ARGS \
        --use_multiple_shooting \
        --seg_length ${SEG} \
        --seg_length_start ${SEG_LENGTH_START} \
        --seg_length_schedule_epochs ${SEG_SCHEDULE_EPOCHS} \
        --shooting_noise_std ${SHOOTING_NOISE_STD} \
        --rollout_loss_weight ${ROLLOUT_LOSS_WEIGHT} \
        --rollout_loss_type ${ROLLOUT_LOSS_TYPE} \
        --rollout_num_ensemble ${ROLLOUT_NUM_ENSEMBLE} \
        --rollout_energy_beta ${ROLLOUT_ENERGY_BETA} \
        --rollout_uncertainty_weight ${ROLLOUT_UNCERTAINTY_WEIGHT} \
        --rollout_every ${ROLLOUT_EVERY} \
        --rollout_warmup_epochs ${ROLLOUT_WARMUP_EPOCHS} \
        --rollout_window ${ROLLOUT_WINDOW} \
        --continuity_loss_weight ${CONTINUITY_LOSS_WEIGHT} \
        --init_loss_weight ${INIT_LOSS_WEIGHT} \
        --initial_state_mode ${INITIAL_STATE_MODE} \
        --seed 0
done
