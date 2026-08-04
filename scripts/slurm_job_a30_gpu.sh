#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=amortized_state_flow
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:a30:[1-4]":
#SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-gpu

export PYTHONNOUSERSITE=1
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

for FLOW_TYPE in realnvp nsf fm; do
    echo "Training AmortizedStateFlow: flow_type=${FLOW_TYPE}, seg_len=25"
    $PYTHON -m experiments.lorenz96_state_estimation.train_amortized_state_flow \
        --train \
        --flow_type "${FLOW_TYPE}" \
        --encoder_type transformer \
        --seg_len 25 \
        --N 12 \
        --obs_interval 2 \
        --obs_std 0.5 \
        --batch_size 32 \
        --learning_rate 5e-4 \
        --grad_norm 1.0 \
        --patience 10
done

