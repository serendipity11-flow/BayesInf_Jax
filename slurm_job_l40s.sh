#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mm
#SBATCH --partition=l40s
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=72:00:00
 
# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:l40s:[1-4]":
##SBATCH --gres=gpu:l40s:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=l40s-cpu

source $HOME/myenv_312/bin/activate
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$PYTHONPATH
python3.12 -c "import sys; print(sys.path)"

cd /home/zhangyi/bayesinf_jax/bayesinf/models/

srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --test --flow_type='FM' --dt0=0.05 --t1=1.0 --noise_std=0.01 --force --learning_rate=0.01 --grad_norm=1.0 --patience=2000 --max_epochs=20000