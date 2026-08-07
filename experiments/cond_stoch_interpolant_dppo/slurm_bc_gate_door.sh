#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_bc_gate
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/results_optimal_control/door/si_analytic_bc_gate/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
OUTPUT_DIR="$PROJECT_ROOT/results/results_optimal_control/door/si_analytic_bc_gate"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-cache-"${SLURM_JOB_ID}"

PYTHON=""
for candidate in \
    "$HOME/myenv_312/bin/python" \
    /usr/bin/python3.12 \
    /usr/local/bin/python3.12
do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "No working Python 3.12 interpreter found on $(hostname)" >&2
    exit 127
fi

echo "job_id=$SLURM_JOB_ID host=$(hostname) python=$PYTHON"
nvidia-smi || true

MANIFEST=experiments/cond_stoch_interpolant_dppo/config/evaluation_door_bc_gate.json
THRESHOLD="${CSI_BC_SUCCESS_THRESHOLD:-0.50}"

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
  --manifest "$MANIFEST"

"$PYTHON" -c '
import csv, sys
path, threshold = sys.argv[1], float(sys.argv[2])
rows = list(csv.DictReader(open(path, newline="")))
if len(rows) != 1:
    raise SystemExit(f"expected one BC gate row, found {len(rows)}")
success = float(rows[0]["success_rate_mean_mean"])
print(f"CSI BC success={success:.3f}; required={threshold:.3f}")
if success < threshold:
    raise SystemExit("CSI BC success gate failed; DPPO will not launch")
' "$OUTPUT_DIR/final_results.csv" "$THRESHOLD"
