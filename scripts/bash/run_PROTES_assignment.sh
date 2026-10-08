#!/usr/bin/env bash
#SBATCH --job-name=protes_eels
#SBATCH --partition=c32m384
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=protes_%A_%a.out
#SBATCH --error=protes_%A_%a.err

set -euo pipefail

# Submit from any directory: bash /path/to/project/scripts/bash/run_PROTES_assignment.sh
# Bootstrap on the login node so compute nodes do not need internet access.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export PROTES_PROJECT_ROOT="$(cd -- "$script_dir/../.." && pwd)"
    cd -- "$PROTES_PROJECT_ROOT"
    command -v julia >/dev/null || { echo 'Julia is required in PATH.' >&2; exit 1; }
    py="${PROTES_PYTHON:-python3}"
    "$py" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 is required"'
    if [[ ! -d PROTES/.git ]]; then
        git clone https://github.com/anabatsh/PROTES.git PROTES
        git -C PROTES checkout --detach 06d447845dd26ffccbc8e95c7545fa1d2617940e
    fi
    [[ "$(git -C PROTES rev-parse HEAD)" == 06d447845dd26ffccbc8e95c7545fa1d2617940e ]] || {
        echo 'Unexpected PROTES revision; the experiment requires the pinned revision.' >&2; exit 1;
    }
    # Source copies are tracked in the parent project because PROTES/ is ignored.
    cp scripts/protes/run_assignment.py scripts/protes/prepare_assignment.jl \
       scripts/protes/requirements-assignment.txt scripts/protes/README-assignment.md \
       scripts/protes/test_assignment.py PROTES/
    [[ -x PROTES/.venv/bin/python ]] || "$py" -m venv PROTES/.venv
    PROTES/.venv/bin/python -m pip install -r PROTES/requirements-assignment.txt
    JULIA_NUM_THREADS=1 julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
    if command -v sbatch >/dev/null 2>&1; then
        exec sbatch --export=ALL --chdir="$PROTES_PROJECT_ROOT" "${BASH_SOURCE[0]}"
    fi
fi

project_root="${PROTES_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -f "$project_root/PROTES/run_assignment.py" ]] || {
    echo 'Launch with bash scripts/bash/run_PROTES_assignment.sh from a checked-out project.' >&2; exit 1;
}
cd -- "$project_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true

# Each array task compares 50 instances of one size/objective; every optimizer
# process and its Julia child are pinned to ONE allowed logical CPU by Python.
# Per-instance EELS measurements are taken on this same server, not extrapolated.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    index="$SLURM_ARRAY_TASK_ID"
    n=$((12 + index % 5))
    objective=linear
    (( index < 5 )) || objective=quadratic
    exec srun --cpu-bind=threads --ntasks=1 --cpus-per-task=1 \
        PROTES/.venv/bin/python PROTES/run_assignment.py \
        --sizes "$n" --instances 1:50 --objectives "$objective" \
        --output "data/protes_comparison/server/${objective}_${n}"
else
    exec PROTES/.venv/bin/python PROTES/run_assignment.py \
        --output data/protes_comparison/server/sequential
fi
