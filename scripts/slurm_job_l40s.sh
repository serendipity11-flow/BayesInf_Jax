#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=pointwise_state_flow
#SBATCH --partition=l40s
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:l40s:[1-4]":
##SBATCH --gres=gpu:l40s:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=l40s-cpu

source $HOME/myenv_312/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$PYTHONPATH
python3.12 -c "import sys; print(sys.path)"

for FLOW_TYPE in fm realnvp nsf; do
    echo "Training PointwiseStateFlow: flow_type=${FLOW_TYPE}, encoder=transformer, seg_len=25"
    python3.12 -m bayesinf.applications.state_estimation.train_pointwise_state_flow \
        --train \
        --flow_type "${FLOW_TYPE}" \
        --encoder_type transformer \
        --seg_len 25 \
        --N 12 \
        --obs_interval 2 \
        --obs_std 0.5 \
        --learning_rate 1e-3 \
        --grad_norm 1.0 \
        --patience 30
done
