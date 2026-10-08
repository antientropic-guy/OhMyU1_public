#!/usr/bin/env bash
#SBATCH --job-name=set_partitioning
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=set_partitioning_%j.out
#SBATCH --error=set_partitioning_%j.err
set -euo pipefail
# Full run: light tiles, entropic objective, 250 instances x 2 pairs = 1000 solves.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export PARTITION_PROJECT_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    if command -v sbatch >/dev/null 2>&1; then
        exec sbatch --export=ALL --chdir="$PARTITION_PROJECT_ROOT" "${BASH_SOURCE[0]}" "$@"
    fi
fi
project_root="${PARTITION_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/run_SetPartitioning.jl" ]] || {
    echo 'Launch with: bash scripts/bash/run_SetPartitioning.sh' >&2
    exit 1
}
cd -- "$project_root"
command -v julia >/dev/null 2>&1 || { echo 'Julia must be in PATH.' >&2; exit 1; }
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${JULIA_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GKSwstype=100
julia --startup-file=no --project="$project_root" -e 'using Pkg; Pkg.instantiate()'
exec julia --startup-file=no --project="$project_root" --threads="$JULIA_NUM_THREADS" \
    "$project_root/scripts/run_SetPartitioning.jl" "$@"
