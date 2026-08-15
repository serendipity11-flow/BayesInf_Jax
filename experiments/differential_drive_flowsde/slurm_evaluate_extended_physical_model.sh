#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_phys8_complex
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
DATA="$PROJECT_ROOT/data/differential_drive/smooth_terrain_navigation_extended_state_postfix.h5"
FLOWSDE="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/joint_stage"
FLOW_CPU="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_cpu/closed_loop_metrics.json"
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex"
EVALUATION="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/evaluation"
MODEL="$OUTPUT/extended_physical_model.npz"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-phys8-${SLURM_JOB_ID}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$OUTPUT" "$EVALUATION" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.extended_physical_model \
    --data "$DATA" --output "$MODEL" --ridge 1e-3

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.compare_open_loop_models \
    --data "$DATA" --result "$FLOWSDE" --physical-model "$MODEL" \
    --output "$EVALUATION" --episodes 10 --horizon 100

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
    --output "$OUTPUT" --extended-model "$MODEL" --task-set complex --tasks 3 \
    --max-steps 6000 --horizon 100 --samples 256 --iterations 3 \
    --temperature .35 --sigma-v .7 --sigma-omega 1.5 --goal-tolerance .35 \
    --report-every 50 --save-samples --seed 47

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.plot_closed_loop_comparison \
    --flowsde "$FLOW_CPU" \
    --physical "$OUTPUT/nonlinear_physical_mppi_metrics.json" \
    --output "$EVALUATION" \
    --flowsde-hardware "CPU (4 cores)" \
    --physical-hardware "CPU (4 cores)"
