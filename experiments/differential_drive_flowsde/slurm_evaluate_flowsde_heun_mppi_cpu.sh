#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flow_heun_mppi
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_cpu/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_cpu/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
RESULT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/joint_stage"
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_cpu"
EVALUATION="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/evaluation"
PHYSICAL="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex/nonlinear_physical_mppi_metrics.json"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-heun-mppi-${SLURM_JOB_ID}"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$OUTPUT" "$EVALUATION" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_closed_loop \
    --result "$RESULT" --output "$OUTPUT" --task-set complex --tasks 3 \
    --max-steps 6000 --mpc-horizon 100 --candidates 256 --elites 32 \
    --cem-iterations 3 --optimizer mppi --temperature .35 \
    --goal-tolerance .35 --report-every 50 --seed 47

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.plot_closed_loop_comparison \
    --flowsde "$OUTPUT/closed_loop_metrics.json" \
    --physical "$PHYSICAL" --output "$EVALUATION" \
    --flowsde-hardware "CPU (4 cores)" \
    --physical-hardware "CPU (4 cores)"
