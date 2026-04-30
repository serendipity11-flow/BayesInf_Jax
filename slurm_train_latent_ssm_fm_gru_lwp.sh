#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=latent_ssm_fm_lwp
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de

export PYTHONNOUSERSITE=1
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH

if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12)
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

for SEG_LEN in 5 10 20 25; do
    echo "Training latent SSM flow matching: seg_len=${SEG_LEN}"
    "$PYTHON" -m bayesinf.applications.state_estimation.train_latent_ssm_flow \
        --train \
        --flow_type fm \
        --fix_ssm \
        --seg_len "$SEG_LEN" \
        --max_rho 0.95 \
        --encoder_type gru \
        --local_window \
        --local_window_prev_only \
        --lambda_ent 0.01 \
        --N 12 \
        --obs_interval 2 \
        --obs_std 0.5 \
        --obs_df 5.0 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --grad_norm 1.0 \
        --patience 50 \
        --max_epochs 2000
done
