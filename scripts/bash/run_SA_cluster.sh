#!/usr/bin/env bash
#SBATCH --job-name=sa_mps_comparison
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=data/sa_cluster_v1/results/logs/%A_%a.out
#SBATCH --error=data/sa_cluster_v1/results/logs/%A_%a.err
set -euo pipefail
for arg in "$@"; do
    case "$arg" in
        --budgets=*|--count=*|--output=*) ;;
        *) echo 'Optional arguments: --budgets=10,50,200 --count=40 --output=PATH' >&2; exit 1 ;;
    esac
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export SA_CLUSTER_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    cd -- "$SA_CLUSTER_ROOT"
    [[ -f data/sa_cluster_v1/manifest.jld2 ]] || { echo 'Missing frozen input snapshot.' >&2; exit 1; }
    command -v julia >/dev/null || { echo 'Julia must be in PATH.' >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'Slurm sbatch must be in PATH.' >&2; exit 1; }
    # Install once before submission; do not run optimization on the login node.
    export JULIA_PKG_PRECOMPILE_AUTO=0
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
    julia --startup-file=no --project=scripts/sa_cluster -e 'using Pkg; Pkg.instantiate()'
    mkdir -p data/sa_cluster_v1/results/logs
    exec sbatch --export=ALL --chdir="$SA_CLUSTER_ROOT" "$script_dir/run_SA_cluster.sh" "$@"
fi
project_root="${SA_CLUSTER_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/sa_cluster/run.jl" ]] || { echo 'Project root not found.' >&2; exit 1; }
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || { echo 'Expected one of four array workers.' >&2; exit 1; }
cd -- "$project_root"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
# Each job has a private result dictionary. Methods are sequential on its CPU.
exec srun --cpu-bind=cores julia --startup-file=no --compiled-modules=existing \
    --project=scripts/sa_cluster --threads=1,0 scripts/sa_cluster/run.jl \
    --job="$SLURM_ARRAY_TASK_ID" "$@"
