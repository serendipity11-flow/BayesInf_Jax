#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_mppi_smooth
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j-aloha-mppi-smoothness.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"
PROJECT=/home/zhangyi/bayesinf_jax
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=$PROJECT:$VENV_SITE:${PYTHONPATH:-}
if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, pandas, pyarrow, mujoco' >/dev/null 2>&1; then
  PYTHON=$HOME/myenv_312/bin/python
else
  PYTHON=$(command -v python3.12 || true)
fi
if [[ -z "${PYTHON:-}" ]]; then echo "Python 3.12 unavailable" >&2; exit 2; fi

cd "$PROJECT"
CONFIG=results/results_optimal_control/aloha_mujoco/flow_sde_small/aloha_flow_sde_heun_seg16_noise0p002_seed0_config.json
ROOT=results/results_optimal_control/aloha_mujoco/flow_sde_small/smoothness_sweep_risk1_particles8_h8_noise0p01
COMMON=(
  --flow-config "$CONFIG" --episode-scope all --tasks 80
  --horizon 8 --particles 8 --risk-weight 1.0 --plant-noise-std 0.01
  --control-steps 10 --candidates 32 --iterations 2
  --exploration-fraction 0.04 --temperature 0.5
  --action-weight 1e-2 --terminal-weight 10.0 --velocity-weight 0.1
  --resume --skip-episode-plots
)

"$PYTHON" -m experiments.aloha_mujoco_flow.compare_mppi \
  "${COMMON[@]}" --output "$ROOT/exact_baseline" \
  --controller exact_mujoco --smoothness-weight 0.0

for WEIGHT in 0 0.01 0.03 0.1 0.3 1.0; do
  TAG=${WEIGHT/./p}
  "$PYTHON" -m experiments.aloha_mujoco_flow.compare_mppi \
    "${COMMON[@]}" --output "$ROOT/weight${TAG}" \
    --controller flow_sde --smoothness-weight "$WEIGHT"
done

"$PYTHON" evaluation/plot_aloha_mppi_smoothness_sweep.py "$ROOT"
