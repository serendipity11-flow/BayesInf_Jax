#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_encoder_flows
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"
mkdir -p results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# The Python 3.12 venv currently contains both CUDA-12 and CUDA-13 cuDNN
# distributions. On the cluster's 570 driver that makes JAX 0.10 select an
# incompatible cuDNN. The established user-site Python has the CUDA-12-only
# JAX 0.5 stack and is compatible with the L40S nodes.
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON=/usr/bin/python3

echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" - <<'PY'
import wandb
if not wandb.api.api_key:
    raise RuntimeError("W&B is not logged in. Run `wandb login` before submitting.")
print("W&B authentication is configured")
PY

"$PYTHON" -m experiments.MEMS_end_of_line.train_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --output-dir results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison \
    --models lstm_bn_flow lstm_flow transformer_flow \
    --epochs 10000 \
    --patience 10000 \
    --batch-size 200 \
    --learning-rate 1e-4 \
    --kernel-gamma 0.1 \
    --max-explr-epoch 2000 \
    --lr-decay 0.8 \
    --lr-decay-epochs 2000 \
    --lr-scheduler \
    --min-learning-rate-ratio 0.01 \
    --width 64 \
    --encoder-layers 2 \
    --transformer-heads 4 \
    --l2-coefficient 1e-5 \
    --diffusion-steps 100 \
    --flow-steps 100 \
    --seed 0 \
    --no-signal-standardization \
    --log-every 25 \
    --wandb-mode online \
    --wandb-project MEMS_end_of_line \
    --wandb-group mems-raw-lstm-bn-vs-no-bn-vs-transformer-10k-seed0

"$PYTHON" -m experiments.MEMS_end_of_line.evaluate_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison \
    --models lstm_bn_flow lstm_flow transformer_flow \
    --posterior-samples 500 \
    --seed 0

"$PYTHON" -m experiments.MEMS_end_of_line.plot_encoder_pca \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison \
    --seed 0
