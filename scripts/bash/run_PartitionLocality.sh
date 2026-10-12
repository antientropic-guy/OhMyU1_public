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
objective=""
grid=0
for arg in "$@"; do
    case "$arg" in
        --linear|--entropic) [[ -z "$objective" ]] || { echo 'Choose exactly one objective.' >&2; exit 1; }; objective="$arg" ;;
        --grid) grid=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done
[[ -n "$objective" ]] || { echo 'Required: --linear or --entropic; optional: --grid' >&2; exit 1; }
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export LOCALITY_PROJECT_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    if command -v sbatch >/dev/null 2>&1; then
        array='0-6%4'
        [[ "$grid" == 0 ]] || array='0-62%4'
        exec sbatch --array="$array" --export=ALL --chdir="$LOCALITY_PROJECT_ROOT" "${BASH_SOURCE[0]}" "$@"
    fi
fi
project_root="${LOCALITY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/run_PartitionLocality.jl" ]] || { echo 'Project root not found.' >&2; exit 1; }
cd -- "$project_root"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GKSwstype=100
command -v julia >/dev/null 2>&1 || { echo 'Julia must be in PATH.' >&2; exit 1; }
spans=(5 6 8 10 12 15 18)
run_task() {
    local task="$1" span m n count input
    local extra=()
    if [[ "$grid" == 1 ]]; then
        span="${spans[$((task / 9))]}"; m=$((18 + (task % 9) / 3)); n=$((13 + task % 3)); count=15
        extra=(--grid --elements "$m" --n "$n")
    else
        span="${spans[$task]}"; count=40
    fi
    for ((i=1; i<=count; i++)); do
        if [[ "$grid" == 1 ]]; then
            input="data/set_partitioning/locality_grid_v1/instances/w${span}_m${m}_n${n}_i${i}.jld2"
        else
            input="data/set_partitioning/locality_v1/instances/span${span}_i${i}.jld2"
        fi
        [[ -f "$input" ]] || { echo "Missing input: $input" >&2; return 1; }
    done
    julia --startup-file=no --project=. --threads=1 scripts/run_PartitionLocality.jl "$objective" --span "$span" "${extra[@]}"
}
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Use this wrapper via bash; it chooses the correct array size on submission.
    [[ "$grid" == 0 || "${SLURM_ARRAY_TASK_COUNT:-0}" == 63 ]] || { echo 'Grid requires 63 tasks; submit using bash, not bare sbatch.' >&2; exit 1; }
    run_task "$SLURM_ARRAY_TASK_ID"
else
    last=6; [[ "$grid" == 0 ]] || last=62
    for ((task=0; task<=last; task++)); do run_task "$task"; done
fi
