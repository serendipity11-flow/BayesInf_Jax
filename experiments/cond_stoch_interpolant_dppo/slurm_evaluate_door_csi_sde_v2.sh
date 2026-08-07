#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_sde_v2_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_sde_v2_evaluation/slurm-%j.out
set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax; cd "$ROOT"
mkdir -p results/results_optimal_control/door/csi_sde_v2_evaluation
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON=""
for candidate in "$HOME/myenv_312/bin/python" /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "No working Python 3.12 interpreter found on $(hostname)" >&2
    exit 127
fi
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
 --manifest experiments/cond_stoch_interpolant_dppo/config/evaluation_door_csi_sde_v2.json
