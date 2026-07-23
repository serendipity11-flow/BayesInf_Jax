#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=aloha_mppi_prisk
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j-aloha-mppi-particle-risk.out

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
if [[ -z "${PYTHON:-}" ]]; then
  echo "Python 3.12 is unavailable on this L40S node" >&2
  exit 2
fi
echo "Using Python: $PYTHON ($($PYTHON --version))"
"$PYTHON" -c 'import jax, equinox, pandas, pyarrow, mujoco; print("dependencies OK")'

cd /home/zhangyi/bayesinf_jax
FLOW_CONFIG=${FLOW_CONFIG:-results/results_optimal_control/aloha_mujoco/flow_sde_small/aloha_flow_sde_heun_seg16_noise0p002_seed0_config.json}
ROOT=${SWEEP_OUTPUT:-results/results_optimal_control/aloha_mujoco/flow_sde_small/particle_risk_sweep}

for RISK in 0 0.25 0.5 1.0; do
  RISK_TAG=${RISK/./p}
  for PARTICLES in 4 8 16 32; do
    OUTPUT="$ROOT/risk${RISK_TAG}_particles${PARTICLES}"
    "$PYTHON" -m experiments.aloha_mujoco_flow.compare_mppi \
      --flow-config "$FLOW_CONFIG" --output "$OUTPUT" \
      --episode-scope all --tasks 80 --controller flow_sde \
      --horizon 8 --particles "$PARTICLES" --risk-weight "$RISK" \
      --plant-noise-std 0.01 --control-steps 10 \
      --candidates 32 --iterations 2 --exploration-fraction 0.04 \
      --temperature 0.5 --action-weight 1e-2 --terminal-weight 10.0 \
      --velocity-weight 0.1 --resume
  done
done

"$PYTHON" evaluation/plot_aloha_mppi_particle_risk_sweep.py "$ROOT"
