#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=eval_flow_sde
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:a30:[1-4]":
##SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-cpu

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH

PYTHON=""
for CANDIDATE in /usr/bin/python3.12 "$HOME/myenv_312/bin/python" /usr/local/bin/python3.12; do
    if [[ -x "$CANDIDATE" ]] && "$CANDIDATE" -c 'import encodings, jax, equinox, optax, diffrax' >/dev/null 2>&1; then
        PYTHON=$CANDIDATE
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: no usable Python 3.12 found. Tried /usr/bin/python3.12, myenv_312/bin/python, /usr/local/bin/python3.12."
    exit 1
fi
echo "Using Python: $PYTHON  ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

N_REALNVP_LAYERS="${N_REALNVP_LAYERS:-2}"
FLOW_TYPE="${FLOW_TYPE:-realnvp}"
N_NICE_LAYERS="${N_NICE_LAYERS:-2}"
NICE_NOISE_SCALE="${NICE_NOISE_SCALE:-0.5}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

BASE_EVAL_ARGS="
    --N 12 --obs_interval 2 --obs_std 0.5
    --horizon 101 --dt 0.02 --F_lorenz 8.0
    --spinup_steps 500 --n_samples 2000
    --obs_noise_type gmm
    --gmm_weights 0.70,0.15,0.15
    --gmm_offsets 0.0,2.5,-2.5
    --gmm_stds 0.50,1.50,1.50
    --d_c 128 --d_hidden_enc 128
    --d_hidden 256 --depth 3
    --d_hidden_diffusion 64
    --time_emb_dim 32
    --n_realnvp_layers ${N_REALNVP_LAYERS} --realnvp_alpha 1.0
    --flow_type ${FLOW_TYPE} --n_nice_layers ${N_NICE_LAYERS}
    --nice_noise_scale ${NICE_NOISE_SCALE}
    --use_custom_solver
    --use_multiple_shooting
    --seed 0
    --n_eval 100 --n_plot_seqs 6
    --n_particles 200 --ci 90.0 --t_hist 0.5"

RESULTS_DIR=/home/zhangyi/bayesinf_jax/results/results_state_estimation

SEG_LIST="${SEG_LIST:-5}"
SEG_LENGTH_START="${SEG_LENGTH_START:-5}"
SOLVER_LIST="${SOLVER_LIST:-heun}"
TEMP_LIST="${TEMP_LIST:-1.8}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-1}"
SHOOTING_NOISE_STD="${SHOOTING_NOISE_STD:-0.01}"
ROLLOUT_LOSS_WEIGHT="${ROLLOUT_LOSS_WEIGHT:-1.0}"
ROLLOUT_LOSS_TYPE="${ROLLOUT_LOSS_TYPE:-energy_score}"
ROLLOUT_NUM_ENSEMBLE="${ROLLOUT_NUM_ENSEMBLE:-20}"
ROLLOUT_ENERGY_BETA="${ROLLOUT_ENERGY_BETA:-1.0}"
ROLLOUT_UNCERTAINTY_WEIGHT="${ROLLOUT_UNCERTAINTY_WEIGHT:-0.5}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-1}"
ROLLOUT_WARMUP_EPOCHS="${ROLLOUT_WARMUP_EPOCHS:-0}"
ROLLOUT_WINDOW="${ROLLOUT_WINDOW:-0}"
CONTINUITY_LOSS_WEIGHT="${CONTINUITY_LOSS_WEIGHT:-0.1}"
INIT_LOSS_WEIGHT="${INIT_LOSS_WEIGHT:-1.0}"

ROBUST_ARGS="
    --shooting_noise_std ${SHOOTING_NOISE_STD}
    --rollout_loss_weight ${ROLLOUT_LOSS_WEIGHT}
    --rollout_loss_type ${ROLLOUT_LOSS_TYPE}
    --rollout_num_ensemble ${ROLLOUT_NUM_ENSEMBLE}
    --rollout_energy_beta ${ROLLOUT_ENERGY_BETA}
    --rollout_uncertainty_weight ${ROLLOUT_UNCERTAINTY_WEIGHT}
    --rollout_every ${ROLLOUT_EVERY}
    --rollout_warmup_epochs ${ROLLOUT_WARMUP_EPOCHS}
    --rollout_window ${ROLLOUT_WINDOW}
    --continuity_loss_weight ${CONTINUITY_LOSS_WEIGHT}
    --init_loss_weight ${INIT_LOSS_WEIGHT}"


ROBUST_TAG=""
SHOOTING_NOISE_TAG=$(printf '%g' "$SHOOTING_NOISE_STD")
ROLLOUT_LOSS_TAG=$(printf '%g' "$ROLLOUT_LOSS_WEIGHT")
CONTINUITY_LOSS_TAG=$(printf '%g' "$CONTINUITY_LOSS_WEIGHT")
if [[ "$SHOOTING_NOISE_TAG" != "0" ]]; then
    ROBUST_TAG="${ROBUST_TAG}_sn${SHOOTING_NOISE_TAG}"
fi
if [[ "$ROLLOUT_LOSS_TAG" != "0" ]]; then
    ROBUST_TAG="${ROBUST_TAG}_rw${ROLLOUT_LOSS_TAG}"
    if [[ "$ROLLOUT_LOSS_TYPE" != "mse" ]]; then
        ROLLOUT_TYPE_TAG="${ROLLOUT_LOSS_TYPE//_/}"
        ROBUST_TAG="${ROBUST_TAG}_${ROLLOUT_TYPE_TAG}${ROLLOUT_NUM_ENSEMBLE}"
        if [[ "$ROLLOUT_LOSS_TYPE" == "energy_score" ]]; then
            ENERGY_BETA_TAG=$(printf '%g' "$ROLLOUT_ENERGY_BETA")
            if [[ "$ENERGY_BETA_TAG" != "1" ]]; then
                ROBUST_TAG="${ROBUST_TAG}b${ENERGY_BETA_TAG}"
            fi
            UNCERTAINTY_WEIGHT_TAG=$(printf '%g' "$ROLLOUT_UNCERTAINTY_WEIGHT")
            if [[ "$UNCERTAINTY_WEIGHT_TAG" != "0.5" ]]; then
                ROBUST_TAG="${ROBUST_TAG}uw${UNCERTAINTY_WEIGHT_TAG}"
            fi
        fi
    fi
    if [[ "$ROLLOUT_EVERY" != "1" ]]; then
        ROBUST_TAG="${ROBUST_TAG}_re${ROLLOUT_EVERY}"
    fi
    if [[ "$ROLLOUT_WARMUP_EPOCHS" != "0" ]]; then
        ROBUST_TAG="${ROBUST_TAG}_rwu${ROLLOUT_WARMUP_EPOCHS}"
    fi
    if [[ "$ROLLOUT_WINDOW" != "0" ]]; then
        ROBUST_TAG="${ROBUST_TAG}_rwnd${ROLLOUT_WINDOW}"
    fi
fi
if [[ "$CONTINUITY_LOSS_TAG" != "0" ]]; then
    ROBUST_TAG="${ROBUST_TAG}_cw${CONTINUITY_LOSS_TAG}"
fi
checkpoint_path() {
    local solver=$1
    local seg=$2
    local mode_tag="${ROBUST_TAG}"
    local model_prefix="RealNVPFlowSDE"
    if [[ "$FLOW_TYPE" == "nice" ]]; then
        model_prefix="NICEFlowSDE"
    fi
    if [[ "$SEG_LENGTH_START" != "$seg" ]]; then
        mode_tag="${mode_tag}_sched${SEG_LENGTH_START}to${seg}"
    fi
    echo "${RESULTS_DIR}/${model_prefix}_N12_h101_obs2_gmm3c_std0.5_pw${solver}_custom${mode_tag}.pt"
}

for SOLVER in $SOLVER_LIST; do
  for SEG in $SEG_LIST; do
    CKPT="${CHECKPOINT_PATH:-$(checkpoint_path "$SOLVER" "$SEG")}"
    if [[ ! -f "$CKPT" ]]; then
        echo "Skipping missing checkpoint: $CKPT"
        continue
    fi
    for TEMP in $TEMP_LIST; do
        echo "========================================"
        echo "  Continuous FlowSDE rollout  provided x0  solver=${SOLVER}  observation_stride=${OBSERVATION_STRIDE}  diffusion_temp=${TEMP}"
        echo "  shooting_noise_std=${SHOOTING_NOISE_STD}  rollout_loss_weight=${ROLLOUT_LOSS_WEIGHT}"
        echo "========================================"
        $PYTHON -m bayesinf.applications.state_estimation.eval_flow_sde_lorenz96 \
            $BASE_EVAL_ARGS \
            $ROBUST_ARGS \
            --solver ${SOLVER} \
            --seg_length ${SEG} \
            --observation_stride ${OBSERVATION_STRIDE} \
            --checkpoint_path ${CKPT} \
            --diffusion_temp ${TEMP}
    done
  done
done
