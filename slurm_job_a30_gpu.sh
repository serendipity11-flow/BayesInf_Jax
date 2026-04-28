#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=latent_ssm_flow
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

source $HOME/myenv_312/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$PYTHONPATH
python3.12 -c "import sys; print(sys.path)"

python3.12 -m bayesinf.applications.state_estimation.train_latent_ssm_flow_lorenz96 \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 50 \
    --max_rho 0.95 \
    --ssm_noise 0.1 \
    --encoder_type gru \
    --local_window \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000 \
    --lambda_ent 0.01 \

python3.12 -m bayesinf.applications.state_estimation.train_latent_ssm_flow_lorenz96 \
    --train \
    --flow_type realnvp \
    --fix_ssm \
    --seg_len 100 \
    --max_rho 0.95 \
    --ssm_noise 0.1 \
    --encoder_type gru \
    --local_window \
    --N 12 \
    --obs_interval 2 \
    --obs_std 0.5 \
    --obs_df 5.0 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --grad_norm 1.0 \
    --patience 50 \
    --max_epochs 2000 \
    --lambda_ent 0.01 \
