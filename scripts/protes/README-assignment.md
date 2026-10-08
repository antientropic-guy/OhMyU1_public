# PROTES versus EELS: binary assignment, one CPU

Upstream revision: `06d447845dd26ffccbc8e95c7545fa1d2617940e`.
The upstream implementation `protes/protes.py` is **not modified**.

## Run on the cluster

From the parent OhMyU1_public checkout:

```bash
bash scripts/bash/run_PROTES_assignment.sh
```

The launcher clones the pinned upstream revision if absent, copies the tracked
integration files from `scripts/protes/`, creates a private Python environment,
installs pinned dependencies, instantiates Julia, and submits ten Slurm tasks
to the existing `c32m384` partition. Each task has one CPU and handles one of
five sizes (12..16) and two objectives (linear/quadratic), 50 test instances.
Python >=3.10 and Julia matching the parent project's environment must be in PATH.
The login node requires package-download access for initial setup; compute nodes
do not. With no Slurm, the same launcher runs the experiment sequentially.

Every run is pinned to one allowed logical CPU; Julia inherits the affinity.
JAX uses CPU only; Julia, BLAS, OpenMP thread counts are one. JAX may create
helper threads, but affinity prevents execution on additional logical CPUs.

## Data and initial exposure

Only `test_random_vectors/c_*_r=50_test.jld2` and
`test_random_bilinear_forms/q_*_r=7_test.jld2` are used.
`prepare_assignment.jl` includes the existing EELS implementation, uses its
exact `pair_seed` and Julia `fill_assignment_vectors!`, and exports its initial
400 binary samples. Python does NOT try to reproduce Julia's RNG.
The binary payload hash is verified in Python, as are all 400 initial costs.
Quadratic data use Julia's column-major order and cost x'Qx, without assuming
that Q is symmetric. Assignment variables retain their original n^2 encoding.

The authors' default PROTES parameters are fixed **before test results**:
rank 5, batch 100, top 10, one Adam update per batch, learning rate 0.05.
Their README recommends these defaults and notes no observed benefit from
changing rank in their experiments; this is NOT proof of optimality on assignment.
The benchmark wrapper `calc/opti/opti_protes.py` uses the same defaults.
The separate QUBO demo uses k=1000 and k_top=5, so defaults are not universal
recommendations for every application. No test-based choice between them is made.
An ordinary dense TT rank is not comparable to the aggregate charge dimensions
of a U(1) model.

PROTES has no dataset argument. We use its existing `sample_ext` hook to expose
the exact initial dataset in a fixed shuffled pass: four batches of 100, each
followed by the unchanged top-10/one-Adam-step rule. The initial incumbent is
the best of all 400 points for both optimizers. This explicit warm-start adapter
is a protocol choice; it is NOT a published PROTES initialization method or a
claim that four updates fully fit the empirical distribution. Initial fitting
is included in the time budget. No initial data are injected afterwards.

No permutation coding, masks, repairs, constraint tensor, or A,b is supplied to
the optimizer. Only the external evaluator checks row and column sums. Valid
costs are monotonically mapped to (2/pi)*atan(cost), invalid candidates get 2.
Since PROTES uses rankings, valid ordering is preserved and all valid points
rank ahead of invalid ones. The reported objective is always the ORIGINAL cost.
Completely invalid batches return [] (the documented upstream skip-update path).
Mixed batches retain the original top-10 rule, even when there are fewer than
10 feasible candidates. Those events and the full feasibility history are saved.
The scorer returns None at the deadline (upstream's documented stop mechanism).
No invalid point can become the reported feasible incumbent.

## Time matching

Each test instance is also run through EELS on the SAME server and CPU, with
400 initial samples, 20 outer iterations, 10000 MPS samples, one sweep, gamma=1,
learning rate 0.05 and link degeneracy 1. Its measured runtime is the PROTES
deadline for that exact instance. This repeats EELS deliberately; old timings
from parallel jobs do not establish one-CPU equivalence. No quality-based tuning
is done on the test set. Measuring runtime is not hyperparameter selection.

Julia is warmed up with one short EELS iteration, JAX with a synthetic binary
objective at the same dimension. Warmup times are saved separately. JAX's
persistent compilation cache is enabled; any residual compilation during the
actual run counts towards its deadline. Model initialization, training,
sampling, feasibility tests and score evaluation are inside the measured run.
File loading, export, environment startup and explicit warmup are outside.
The deadline is cooperative: a current JAX update/sample can overrun it. No
scores from a newly arriving batch are accepted after the deadline; any overrun
is measured explicitly. Time ratios should be checked after the cluster run.

## Outputs and resume

`data/protes_comparison/server/<objective>_<n>/` contains:

* `<objective>_<n>_<i>.json`: best feasible original objective, incumbent, curves,
  generated/feasible counts, unique new evaluations, wall/CPU time, deadline
  overrun, EELS reference objective/time, versions, seeds and input hashes.
* `*_summary.csv`: direct EELS/PROTES time and objective comparison.
* `inputs/*/eels.jld2`: original EELS SolverStatistics, diagnostics and parameters.
* `inputs/*/eels_timing.tsv`, `eels_curve.csv`, raw exported input and hashes.
* `*_warmup.json`, `experiment_manifest.json`: warmup and provenance.

Completed results are reused only with matching signatures. A different host,
environment or experiment source requires a new output directory. Interrupted
groups leave a `.running` directory; remove it only after verifying no job is
still writing. The full source scripts are tracked under `scripts/protes/`
because the downloaded PROTES directory is intentionally gitignored.

## Interpretation

This is **data-warm-started, unstructured binary PROTES with a feasibility
evaluator**, not the structural-indicator constrained PROTES from its paper.
A finite Ax=b indicator CAN be represented as a TT, potentially with impractical
ranks. Simply exposing feasible observations does not enforce their constraints.
If feasibility collapses, the experiment primarily measures that limitation;
it does not establish superiority over a well-configured constrained PROTES.
Do not retune on the test outcomes. A future study of initial fitting/ranks
should use training instances and be frozen before a fresh evaluation.

Local calibration (n=12, instance 1, one logical CPU, 2026-10-07):
EELS linear 48.0988 s; quadratic 49.3800 s. First PROTES runs took 48.1121 s
and 49.3936 s (see exact saved CSV). Neither generated a new feasible point;
their returned solutions were the initial incumbents. This is an observed
limitation of this default initialization, not a tuned benchmark result.

## Assignment-size and parameter-sensitivity diagnostics

`examples/PROTES_assignment_scaling.ipynb` contains the rank derivation,
executed sensitivity results, and full-indicator storage comparison. These
diagnostics use newly generated synthetic instances, separate from the paper's
held-out test files. The same initial-data and objective hashes are checked
against the earlier size probe. They are not a fresh SOTA evaluation.

Run with the PROTES Python environment from the repository root:

```sh
python scripts/protes/assignment_sensitivity.py --stage short
python scripts/protes/assignment_sensitivity.py --stage long
python scripts/protes/check_sparse_support.py
julia --project=. --threads=1 scripts/protes/assignment_memory.jl
python scripts/protes/snapshot_assignment_sensitivity.py
```

The short stage uses n=4..10, one matched seed, both objectives, 5 seconds;
the long stage uses n=8, three matched seeds, both objectives, 50 seconds.
All four configurations run at both stages. Parameters are defined explicitly
in `assignment_sensitivity.py`: increased rank, smaller learning rate/more Adam
steps, larger batch, and 20 initial-data passes. Initial fitting counts toward
the budget. Saved JSON files permit resume without repeating completed trials.
To intentionally repeat on another machine, use a separate checkout/output
directory rather than mixing cached results from different hardware.

Compact results and actual Julia object sizes are saved under `examples/` so
the notebook can render without ignored raw files. The raw history is under
`data/protes_comparison/scaling`. Sparse-core checks exercise both unchanged
upstream entry points, including a dense variable-rank positive control.

## Full indicators and paired U(1) sampling

The notebook also separates a full-indicator sampling benchmark from a
data-initialized optimization benchmark. Full dense PROTES cores are ACTUALLY
constructed only for n=4..10 (both upstream implementations); larger dense
storage numbers remain analytical estimates. Full U(1) objects are built to
n=16. The uniform U(1) model is put into exact right-canonical form using
backward suffix counts, then sampled with the unchanged library sampler.

```sh
python scripts/protes/export_scaling_inputs.py
python scripts/protes/run_full_u1_benchmarks.py
python scripts/protes/run_full_protes_benchmarks.py
python scripts/protes/run_julia_single_cpu.py scripts/protes/benchmark_data_u1.jl short
python scripts/protes/run_julia_single_cpu.py scripts/protes/benchmark_data_u1.jl long
python scripts/protes/snapshot_full_comparison.py
python scripts/protes/update_full_comparison_notebook.py
```

Run these sequentially, never simultaneously on the same CPU. The Julia
launcher pins the child to the same first allowed logical CPU as PROTES. The
full U(1) script accepts a starting n to append larger sizes in a fresh process;
ensure the existing CSV ends immediately before that size. Full U(1) at n=16
requires substantial additional memory for Julia metadata and traversal.

The time-limited EELS diagnostic retains the paper's 400 training points,
10000 samples/outer iteration, one training sweep, gamma=1 and learning rate
0.05; it replaces the fixed 20-iteration cap by the 5/50-second deadline.
Generation is checked in batches of 100 to limit deadline overrun. Its input
arrays are byte-identical to the previous PROTES diagnostic, not regenerated
from a Julia interpretation of the same seed. U(1) still has A,b, whereas
data-only PROTES does not: this distinction is explicit in the notebook.
