#!/usr/bin/env bash
#SBATCH --job-name=tt_time_budget
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=380000M
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=data/time_budget_v1/logs/%A_%a.out
#SBATCH --error=data/time_budget_v1/logs/%A_%a.err
set -euo pipefail
for arg in "$@"; do
    case "$arg" in
        --budgets=*|--output=*) ;;
        *) echo 'Usage: bash scripts/bash/run_time_budget.sh [--budgets=10,50,200] [--output=PATH]' >&2; exit 1 ;;
    esac
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export TIME_BUDGET_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    cd -- "$TIME_BUDGET_ROOT"
    command -v julia >/dev/null
    command -v python3 >/dev/null
    command -v sbatch >/dev/null
    export JULIA_PKG_PRECOMPILE_AUTO=0
    # Prepare once, never activate/instantiate from the 128 worker processes.
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); using OhMyU1, JLD2'
    julia --startup-file=no --project=. --threads=1,0 scripts/time_budget/run.jl --prepare "$@"
    mkdir -p data/time_budget_v1/logs
    exec sbatch --export=ALL --chdir="$TIME_BUDGET_ROOT" "$script_dir/run_time_budget.sh" "$@"
fi
cd -- "${TIME_BUDGET_ROOT:-$SLURM_SUBMIT_DIR}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
output=data/time_budget_v1
for arg in "$@"; do [[ "$arg" != --output=* ]] || output="${arg#--output=}"; done
mkdir -p "$output/monitor"
export TIME_BUDGET_OUTPUT="$output"
scontrol show job "$SLURM_JOB_ID"
exec srun --ntasks=32 --cpus-per-task=1 --cpu-bind=cores bash -c '
    exec python3 scripts/time_budget/monitor.py \
        "$TIME_BUDGET_OUTPUT/monitor/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_PROCID}.csv" \
        julia --startup-file=no --compiled-modules=existing --project=. --threads=1,0 --gcthreads=1 \
        scripts/time_budget/run.jl --job="$SLURM_ARRAY_TASK_ID" \
        --worker="$SLURM_PROCID" --workers=32 "$@"
' _ "$@"
