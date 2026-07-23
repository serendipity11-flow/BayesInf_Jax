#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_mujoco_sweep
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de

set -euo pipefail
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:${PYTHONPATH:-}

if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, pandas, pyarrow, mujoco' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || true)
fi
if [ -z "$PYTHON" ]; then
    echo "Usable Python environment not found" >&2
    exit 1
fi

cd /home/zhangyi/bayesinf_jax
FLOW_CONFIG=${FLOW_CONFIG:-results/results_optimal_control/aloha_mujoco/flow_sde_small/aloha_flow_sde_heun_seg16_noise0p002_seed0_config.json}
OUTPUT=${SWEEP_OUTPUT:-results/results_optimal_control/aloha_mujoco/flow_sde_small/all_episode_mppi}
HORIZONS=${HORIZONS:-2,4,6,8,10,12}
NOISE_STD=${PLANT_NOISE_STD:-0.01}
TASKS=${TASKS:-80}

"$PYTHON" -m experiments.aloha_mujoco_flow.compare_mppi \
    --flow-config "$FLOW_CONFIG" \
    --output "$OUTPUT" \
    --horizon-grid "$HORIZONS" \
    --episode-scope all \
    --plant-noise-std "$NOISE_STD" \
    --tasks "$TASKS" \
    --resume \
    --control-steps 10 \
    --candidates 32 \
    --particles 4 \
    --iterations 2 \
    --exploration-fraction 0.04 \
    --temperature 0.5 \
    --action-weight 1e-2 \
    --terminal-weight 10.0 \
    --velocity-weight 0.1
