#!/usr/bin/env bash
#SBATCH --job-name=add_locality
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=380000M
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=data/set_partitioning/locality_grid_v1/results_add_transfer_v3/logs/%A_%a.out
#SBATCH --error=data/set_partitioning/locality_grid_v1/results_add_transfer_v3/logs/%A_%a.err
set -euo pipefail
resume_from=""
expected_completed=""
if [[ $# != 0 ]]; then
    [[ $# == 4 && "$1" == --resume-from && "$3" == --expected-completed ]] || {
        echo 'Usage: bash scripts/bash/run_ADD_locality_transfer.sh [--resume-from HASH --expected-completed COUNT]' >&2; exit 1;
    }
    resume_from="$2"
    expected_completed="$4"
    [[ "$resume_from" =~ ^[0-9a-f]{16}$ && "$expected_completed" =~ ^[0-9]+$ ]] || exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export ADD_LOCALITY_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    cd -- "$ADD_LOCALITY_ROOT"
    command -v julia >/dev/null
    command -v sbatch >/dev/null
    [[ -f data/set_partitioning/locality_grid_v1/manifest.jld2 ]]
    export JULIA_PKG_PRECOMPILE_AUTO=0
    if [[ -n "$resume_from" ]]; then
        # Recovery validates saved results only; it performs no optimization or Pkg operations.
        julia --startup-file=no --compiled-modules=existing --project=. --threads=1,0 \
            scripts/resume_ADD_locality_transfer.jl "$resume_from" "$expected_completed"
    else
        julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
    fi
    mkdir -p data/set_partitioning/locality_grid_v1/results_add_transfer_v3/logs
    exec sbatch --export=ALL --chdir="$ADD_LOCALITY_ROOT" "$script_dir/run_ADD_locality_transfer.sh"
fi
cd -- "${ADD_LOCALITY_ROOT:-$SLURM_SUBMIT_DIR}"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100
# Each process owns its instance file and both sequential arms. No shared dictionaries.
exec srun --ntasks=32 --cpus-per-task=1 --cpu-bind=cores bash -c '
    echo "Starting Julia worker $SLURM_PROCID in array task $SLURM_ARRAY_TASK_ID on $(hostname)"
    exec julia --startup-file=no --compiled-modules=existing --project=. --threads=1,0 \
        scripts/run_ADD_locality_transfer.jl --job="$SLURM_ARRAY_TASK_ID" \
        --worker="$((SLURM_PROCID + 1))" --workers=32
'
