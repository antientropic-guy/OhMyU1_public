#!/usr/bin/env bash
#SBATCH --job-name=sa_penalty_comparison
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=380000M
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=data/sa_penalty_v1/results/logs/%A_%a.out
#SBATCH --error=data/sa_penalty_v1/results/logs/%A_%a.err
set -euo pipefail
for arg in "$@"; do
    case "$arg" in
        --budgets=*|--count=*|--output=*) ;;
        *) echo 'Optional arguments: --budgets=10,50,200 --count=40 --output=PATH' >&2; exit 1 ;;
    esac
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export SA_PENALTY_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    cd -- "$SA_PENALTY_ROOT"
    [[ -f data/sa_cluster_v2/manifest.jld2 ]] || { echo 'Missing frozen input snapshot.' >&2; exit 1; }
    command -v julia >/dev/null || { echo 'Julia must be in PATH.' >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'Slurm sbatch must be in PATH.' >&2; exit 1; }
    # Install once before submission; do not run optimization on the login node.
    export JULIA_PKG_PRECOMPILE_AUTO=0
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
    julia --startup-file=no --project=scripts/sa_cluster -e 'using Pkg; Pkg.instantiate()'
    mkdir -p data/sa_penalty_v1/results/logs
    exec sbatch --export=ALL --chdir="$SA_PENALTY_ROOT" "$script_dir/run_SA_penalty.sh" "$@"
fi
project_root="${SA_PENALTY_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/scripts/sa_cluster/run_penalty.jl" ]] || { echo 'Project root not found.' >&2; exit 1; }
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || { echo 'Expected one of four array workers.' >&2; exit 1; }
cd -- "$project_root"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
# Record the actual node topology and allocation for interpreting timings.
scontrol show job "$SLURM_JOB_ID"
scontrol show node "$SLURMD_NODENAME"
# One srun step starts 32 isolated Julia processes on 32 physical cores.
# SLURM_PROCID assigns a private dictionary; no shared mutable results exist.
exec srun --ntasks=32 --cpus-per-task=1 --cpu-bind=cores bash -c '
    exec julia --startup-file=no --compiled-modules=existing \
        --project=scripts/sa_cluster --threads=1,0 scripts/sa_cluster/run_penalty.jl \
        --job="$SLURM_ARRAY_TASK_ID" --worker="$((SLURM_PROCID + 1))" --workers=32 "$@"
' _ "$@"
