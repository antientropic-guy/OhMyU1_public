#!/usr/bin/env bash
#SBATCH --job-name=partition_locality
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-6%4
#SBATCH --output=partition_locality_%A_%a.out
#SBATCH --error=partition_locality_%A_%a.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export LOCALITY_PROJECT_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    if command -v sbatch >/dev/null 2>&1; then
        exec sbatch --export=ALL --chdir="$LOCALITY_PROJECT_ROOT" "${BASH_SOURCE[0]}"
    fi
fi
project_root="${LOCALITY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/run_PartitionLocality.jl" ]] || {
    echo 'Launch from the project: bash scripts/bash/run_PartitionLocality.sh' >&2; exit 1;
}
cd -- "$project_root"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GKSwstype=100
command -v julia >/dev/null 2>&1 || { echo 'Julia must be in PATH.' >&2; exit 1; }
# Use the generated inputs. Missing data must never silently regenerate on a worker.
spans=(5 6 8 10 12 15 18)
for span in "${spans[@]}"; do
    for i in {1..40}; do
        [[ -f "data/set_partitioning/locality_v1/instances/span${span}_i${i}.jld2" ]] || {
            echo "Missing input span=$span i=$i; copy data/set_partitioning/locality_v1/instances first." >&2; exit 1;
        }
    done
done
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    julia --startup-file=no --project=. --threads=1 scripts/run_PartitionLocality.jl --span "${spans[$SLURM_ARRAY_TASK_ID]}"
else
    for span in "${spans[@]}"; do
        julia --startup-file=no --project=. --threads=1 scripts/run_PartitionLocality.jl --span "$span"
    done
fi
