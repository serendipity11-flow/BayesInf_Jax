#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_mj_fsde
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-aloha-mujoco-flow.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
  PYTHON=$HOME/myenv_312/bin/python
else
  PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
if [[ -z "${PYTHON:-}" ]]; then
  echo "Python 3.12 is unavailable on this compute node" >&2
  exit 2
fi
echo "Using Python: $PYTHON ($($PYTHON --version))"
$PYTHON -c 'import jax, mujoco, equinox, pandas, pyarrow; print("dependencies OK")'
cd /home/zhangyi/bayesinf_jax

DATA=${DATA:-data/aloha_mujoco/trajectories.parquet}
OUTPUT=${OUTPUT:-results/results_optimal_control/aloha_mujoco/flow_sde_small}

$PYTHON -m experiments.aloha_mujoco_flow.generate_simulator_data \
  --output "$DATA" --smooth-episodes 48 --nominal-episodes 24 \
  --mppi-episodes 8 --steps 250 --control-dt 0.02

$PYTHON -m bayesinf.applications.optimal_control.train_aloha_flow_sde \
  --data "$DATA" --output "$OUTPUT" --epochs 100 --batch-size 16 \
  --dt 0.02 --width 64 --depth 2 --diffusion-width 64 --flow-layers 2 \
  --time-embedding 16 --window-steps 64 --segment-length 16 --stride 8 \
  --samples-per-epoch 1024 --train-particles 4 --solver heun \
  --state-noise-std 0.002 --control-noise-std 0.005 \
  --uncertainty-weight 0.6 --continuity-weight 1.0 --seed 0

CONFIG=$(find "$OUTPUT" -maxdepth 1 -name '*_config.json' | head -1)
$PYTHON -m bayesinf.applications.optimal_control.eval_aloha_flow_sde \
  --data "$DATA" --config "$CONFIG" --output "$OUTPUT/evaluation"

$PYTHON -m experiments.aloha_mujoco_flow.compare_mppi \
  --data "$DATA" --flow-config "$CONFIG" \
  --output "$OUTPUT/mppi_comparison" --tasks 5 --horizon 8 \
  --candidates 32 --particles 4 --iterations 2 --control-steps 10

echo "Pipeline complete. Flow config: $CONFIG"
