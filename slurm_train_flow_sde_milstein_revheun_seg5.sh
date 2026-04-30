#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=flow_sde_s5_mil_rev
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:a30:1
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
    --n_nice_layers 2
    --use_custom_solver"

for SOLVER in milstein reversible_heun; do
    echo "========================================"
    echo "  NICEFlowSDE  seg_len=5  custom solver  pathwise/${SOLVER}"
    echo "========================================"
    $PYTHON -m bayesinf.applications.state_estimation.train_flow_sde_lorenz96 \
        --train \
        $DATA_ARGS $NOISE_ARGS $ARCH_ARGS $TRAIN_ARGS \
        --solver ${SOLVER} \
        --use_multiple_shooting \
        --seg_length 5 \
        --seed 0
done
