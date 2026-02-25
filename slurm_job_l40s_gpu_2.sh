#!/bin/bash
# Job name:
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=stress
#SBATCH --partition=l40s
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=72:00:00
 
# Request memory per CPU
#SBATCH --mem-per-cpu=64G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:l40s:[1-4]":
#SBATCH --gres=gpu:l40s:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=l40s-gpu

source $HOME/myenv_312/bin/activate
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$PYTHONPATH
python3.12 -c "import sys; print(sys.path)"

cd /home/zhangyi/bayesinf_jax/bayesinf/models/
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.01 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.02 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.05 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.1 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.2 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='FM' --dt0=0.05 --t1=1.0 --force --noise_std=0.5 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSF' --dt0=0.05 --t1=1.0 --force --noise_std=0.01 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSF' --dt0=0.05 --t1=1.0 --force --noise_std=0.02 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSF' --dt0=0.05 --t1=1.0 --force --noise_std=0.05 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSF' --dt0=0.05 --t1=1.0 --force --noise_std=0.1 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSf' --dt0=0.05 --t1=1.0 --force --noise_std=0.2 --learning_rate=0.005 --patience=500
srun --ntasks=1 --exact python3.12 train_parameter_identification_model.py --train --flow_type='NSF' --dt0=0.05 --t1=1.0 --force --noise_std=0.5 --learning_rate=0.005 --patience=500