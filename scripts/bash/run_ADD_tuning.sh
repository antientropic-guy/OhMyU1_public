#!/usr/bin/env bash
#SBATCH --job-name=assignment_add_tuning
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=add_tuning_%A_%a.out
#SBATCH --error=add_tuning_%A_%a.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export ADD_PROJECT_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    command -v sbatch >/dev/null 2>&1 || { echo 'Slurm sbatch must be in PATH.' >&2; exit 1; }
    exec sbatch --export=ALL --chdir="$ADD_PROJECT_ROOT" "$script_dir/run_ADD_tuning.sh" "$@"
fi
project_root="${ADD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/run_ADD_tuning.jl" ]] || { echo 'Project root not found.' >&2; exit 1; }
cd -- "$project_root"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GKSwstype=100
strategies=(best_cost mixed weighted_rank gdf)
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || { echo 'Expected one of four strategy array tasks.' >&2; exit 1; }
# Limit live MPS/solver instances, not just the number of scheduled tasks.
# Each job owns its files; no inter-job synchronization or shared result dict.
exec julia --startup-file=no --project=. --threads="$JULIA_NUM_THREADS" scripts/run_ADD_tuning.jl \
    --strategy "${strategies[$SLURM_ARRAY_TASK_ID]}" --objective both --workers "${ADD_WORKERS:-4}" "$@"
