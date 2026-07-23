#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_fsde_mppi
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de

set -euo pipefail
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:${PYTHONPATH:-}

if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, pandas, pyarrow' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || true)
fi
if [ -z "$PYTHON" ]; then
    echo "Usable Python environment not found" >&2
    exit 1
fi

cd /home/zhangyi/bayesinf_jax
FLOW_DIR=${FLOW_DIR:-results/results_optimal_control/aloha_mobile_cabinet/flow_sde_dense_uw0p8}
CONFIG=${FLOW_CONFIG:-$FLOW_DIR/aloha_flow_sde_heun_seg16_noise0p01_seed0_config.json}
OUTPUT=${MPPI_OUTPUT:-$FLOW_DIR/mppi_comparison}

"$PYTHON" -c 'import jax, pyarrow; print("JAX devices:", jax.devices()); print("pyarrow:", pyarrow.__version__)'
"$PYTHON" -m bayesinf.applications.optimal_control.compare_aloha_flow_sde_physical_mppi \
    --data data/aloha_mobile_cabinet/file-000.parquet \
    --flow-config "$CONFIG" \
    --output "$OUTPUT" \
    --solver heun \
    --history 8 \
    --horizon 10 \
    --task-steps 50 \
    --tasks 3 \
    --start 400 \
    --candidates 128 \
    --particles 4 \
    --iterations 3 \
    --exploration-std 0.25 \
    --temperature 0.3 \
    --risk-weight 0.1 \
    --terminal-weight 5.0 \
    --action-weight 1e-3
