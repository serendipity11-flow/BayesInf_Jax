#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=latent_ssm_flow
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
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($(${PYTHON} --version))"

$PYTHON -m experiments.lorenz96_state_estimation.train_latent_ssm_flow \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 5 \
    --max_rho 0.95 \
    --encoder_type gru \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000

$PYTHON -m experiments.lorenz96_state_estimation.train_latent_ssm_flow \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 10 \
    --max_rho 0.95 \
    --encoder_type transformer \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000

$PYTHON -m experiments.lorenz96_state_estimation.train_latent_ssm_flow \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 15 \
    --max_rho 0.95 \
    --encoder_type transformer \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000

$PYTHON -m experiments.lorenz96_state_estimation.train_latent_ssm_flow \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 20 \
    --max_rho 0.95 \
    --encoder_type transformer \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000

$PYTHON -m experiments.lorenz96_state_estimation.train_latent_ssm_flow \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 25 \
    --max_rho 0.95 \
    --encoder_type transformer \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000
