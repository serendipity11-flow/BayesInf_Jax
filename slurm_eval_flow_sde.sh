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
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

BASE_EVAL_ARGS="
    --N 12 --obs_interval 2 --obs_std 0.5
    --horizon 101 --dt 0.02 --F_lorenz 8.0
    --spinup_steps 500 --n_samples 2000
    --obs_noise_type gmm
    --gmm_weights 0.70,0.15,0.15
    --gmm_offsets 0.0,2.5,-2.5
    --gmm_stds 0.50,1.50,1.50
    --d_c 128 --d_hidden_enc 128
    --d_hidden 64 --depth 2
    --time_emb_dim 32
    --n_nice_layers 2
    --use_custom_solver
    --use_multiple_shooting
    --seed 0
    --n_eval 100 --n_plot_seqs 6
    --n_particles 200 --ci 90.0 --t_hist 0.5"

RESULTS_DIR=/home/zhangyi/bayesinf_jax/results/results_state_estimation

checkpoint_path() {
    local solver=$1
    local seg=$2
    echo "${RESULTS_DIR}/NICEFlowSDE_N12_h101_obs2_gmm3c_std0.5_dc128_dh64_dep2_nl2_seg${seg}_pw${solver}_custom.pt"
}

for SOLVER in euler heun midpoint milstein; do
  for SEG in 5 10 20 25; do
    CKPT=$(checkpoint_path "$SOLVER" "$SEG")
    if [[ ! -f "$CKPT" ]]; then
        echo "Skipping missing checkpoint: $CKPT"
        continue
    fi
    for TEMP in 1.0 0.8 0.6; do
        echo "========================================"
        echo "  NICEFlowSDE whole-series eval  trained_solver=${SOLVER}  seg_length=${SEG}  sampler=${SOLVER}  diffusion_temp=${TEMP}"
        echo "========================================"
        $PYTHON -m bayesinf.applications.state_estimation.eval_flow_sde_lorenz96 \
            $BASE_EVAL_ARGS \
            --solver ${SOLVER} \
            --sampler ${SOLVER} \
            --seg_length ${SEG} \
            --diffusion_temp ${TEMP}
    done
  done
done
