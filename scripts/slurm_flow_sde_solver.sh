#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=flow_sde
#SBATCH --partition=l40s
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=16G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:l40s:[1-4]":
##SBATCH --gres=gpu:l40s:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=l40s-cpu

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
    --d_hidden 64 --depth 2
    --time_emb_dim 32"

TRAIN_ARGS="
    --batch_size 32
    --learning_rate 3e-4 --grad_norm 1.0 --weight_decay 1e-4
    --patience 40 --max_epochs 500
    --n_realnvp_layers 4 --realnvp_alpha 1.0
    --use_custom_solver"

# Multiple-shooting setting for stochastic initial-state inference:
# z0 ~ N(mu_GRU(y_segment), diag(sigma_GRU(y_segment)^2)).
# The robustness terms keep the short anchored windows,
# but expose training to rollout-like off-manifold starts.
SEG_LIST="${SEG_LIST:-5}"
SEG_LENGTH_START="${SEG_LENGTH_START:-5}"
SEG_SCHEDULE_EPOCHS="${SEG_SCHEDULE_EPOCHS:-100}"
SHOOTING_NOISE_STD="${SHOOTING_NOISE_STD:-0.0}"
ROLLOUT_LOSS_WEIGHT="${ROLLOUT_LOSS_WEIGHT:-0.0}"
CONTINUITY_LOSS_WEIGHT="${CONTINUITY_LOSS_WEIGHT:-0.1}"
INIT_LOSS_WEIGHT="${INIT_LOSS_WEIGHT:-1.0}"
INITIAL_STATE_MODES="${INITIAL_STATE_MODES:-inferred provided}"
SOLVER_LIST="${SOLVER_LIST:-euler reversible_heun midpoint milstein adaptive_euler rk4}"

# ---------------------------------------------------------------------------
# RealNVP transition-NLL training, non-Heun solver sweep
# ---------------------------------------------------------------------------
for INITIAL_STATE_MODE in $INITIAL_STATE_MODES; do
for SOLVER in $SOLVER_LIST; do
    if [[ "$SOLVER" == "heun" ]]; then
        echo "Skipping heun because this script is for non-Heun solver runs."
        continue
    fi
    for SEG in $SEG_LIST; do
        echo "========================================"
        echo "  RealNVPFlowSDE  seg_len=${SEG_LENGTH_START}->${SEG}  custom solver  pathwise/${SOLVER}"
        echo "  initial_state_mode=${INITIAL_STATE_MODE}"
        echo "  shooting_noise_std=${SHOOTING_NOISE_STD}  rollout_loss_weight=${ROLLOUT_LOSS_WEIGHT}"
        echo "  continuity_loss_weight=${CONTINUITY_LOSS_WEIGHT}  init_loss_weight=${INIT_LOSS_WEIGHT}"
        echo "========================================"
        $PYTHON -m bayesinf.applications.state_estimation.train_flow_sde_lorenz96 \
            --train \
            $DATA_ARGS $NOISE_ARGS $ARCH_ARGS $TRAIN_ARGS \
            --solver ${SOLVER} \
            --sampler ${SOLVER} \
            --use_multiple_shooting \
            --seg_length ${SEG} \
            --seg_length_start ${SEG_LENGTH_START} \
            --seg_length_schedule_epochs ${SEG_SCHEDULE_EPOCHS} \
            --shooting_noise_std ${SHOOTING_NOISE_STD} \
            --rollout_loss_weight ${ROLLOUT_LOSS_WEIGHT} \
            --continuity_loss_weight ${CONTINUITY_LOSS_WEIGHT} \
            --init_loss_weight ${INIT_LOSS_WEIGHT} \
            --initial_state_mode ${INITIAL_STATE_MODE} \
            --seed 0
    done
done
done
