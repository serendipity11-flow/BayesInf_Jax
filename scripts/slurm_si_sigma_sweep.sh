#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=sigma_si
#SBATCH --partition=l40s
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=16G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:l40s:[1-4]":
#SBATCH --gres=gpu:l40s:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=l40s-gpu

export PYTHONNOUSERSITE=1
cd /home/zhangyi/bayesinf_jax
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($(${PYTHON} --version))"

# Fixed checkpoint family settings. The only swept parameter is sigma_0.
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
    --d_c 128
    --d_hidden 256 --n_heads 4 --depth 4
    --time_emb_dim 32"

COMMON_TRAIN_ARGS="
    --batch_size 32
    --learning_rate 3e-4 --grad_norm 1.0 --weight_decay 1e-4
    --patience 50 --max_epochs 600
    --g0 0.5
    --score_weight 0.25 --lifter_loss_weight 0.1 --score_sigma_floor 0.05
    --s_min 0.02
    --interpolant_type linear
    --wandb"

COMMON_EVAL_ARGS="
    $DATA_ARGS $NOISE_ARGS $ARCH_ARGS
    --g0 0.5
    --score_weight 0.25 --lifter_loss_weight 0.1 --score_sigma_floor 0.05
    --s_min 0.02
    --interpolant_type linear
    --seed 0
    --n_eval 100 --n_plot_seqs 6
    --n_particles 200 --n_steps 25
    --ci 90.0 --t_hist 0.5
    --sampler segmented
    --ms --seg_len 10
    --seg_len_eval 100"

SIGMA_GRID=${SIGMA_GRID:-0.0,0.1,0.25,0.5,1.0}

RESULTS_ROOT=${RESULTS_ROOT:-/home/zhangyi/bayesinf_jax/results/results_state_estimation}
mkdir -p "$RESULTS_ROOT"

for SIGMA in $(echo "$SIGMA_GRID" | tr ',' ' '); do
    OUT_DIR="$RESULTS_ROOT"

    echo "========================================"
    echo "  SI stochasticity sweep  sigma_0=${SIGMA}"
    echo "========================================"

    $PYTHON -m bayesinf.applications.state_estimation.train_si_lorenz96 \
        --train --test \
        $DATA_ARGS $NOISE_ARGS $ARCH_ARGS $COMMON_TRAIN_ARGS \
        --sigma_0 "$SIGMA" \
        --ms --seg_len 10 \
        --sampler segmented --seg_len_eval 10 \
        --n_particles 200 --n_steps 100 \
        --seed 0

    # The trainer formats sigma with a decimal (e.g. sig0.0), not sig0p0.
    CKPT=$(find "$OUT_DIR" -maxdepth 1 -type f -name "SI*_sig${SIGMA}_*.pt" | sort | tail -1)
    if [ -z "${CKPT:-}" ]; then
        SIGMA_TAG=$(printf "%s" "$SIGMA" | sed 's/\./p/g')
        CKPT=$(find "$OUT_DIR" -maxdepth 1 -name "SI*_sig${SIGMA_TAG}_*.pt" | sort | tail -1)
    fi
    if [ -z "${CKPT:-}" ]; then
        echo "Could not find checkpoint in $OUT_DIR" >&2
        exit 1
    fi

    $PYTHON -m bayesinf.applications.state_estimation.eval_si_lorenz96 \
        $COMMON_EVAL_ARGS \
        --sigma_0 "$SIGMA" \
        --ckpt_path "$CKPT" \
        --output_dir "$OUT_DIR"
done
