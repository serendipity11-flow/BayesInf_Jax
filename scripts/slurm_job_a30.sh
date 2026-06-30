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
##SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-cpu

source $HOME/myenv_312/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$PYTHONPATH
python3.12 -c "import sys; print(sys.path)"

python3.12 -m bayesinf.applications.state_estimation.train_pointwise_state_flow \
    --train \
    --flow_type fm \
    --encoder_type transformer \
    --seg_len 5 \
    --learning_rate 1e-3 \
    --grad_norm 1.0 \
    --patience 30 \

python3.12 -m bayesinf.applications.state_estimation.train_pointwise_state_flow \
    --train \
    --flow_type fm \
    --encoder_type transformer \
    --seg_len 10 \
    --learning_rate 1e-3 \
    --grad_norm 1.0 \
    --patience 30 \

python3.12 -m bayesinf.applications.state_estimation.train_pointwise_state_flow \
    --train \
    --flow_type fm \
    --encoder_type transformer \
    --seg_len 15 \
    --learning_rate 1e-3 \
    --grad_norm 1.0 \
    --patience 30 \

python3.12 -m bayesinf.applications.state_estimation.train_pointwise_state_flow \
    --train \
    --flow_type fm \
    --encoder_type transformer \
    --seg_len 20 \
    --learning_rate 1e-3 \
    --grad_norm 1.0 \
    --patience 30 \
