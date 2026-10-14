# Generic penalty SA on the frozen cluster benchmark

This is a new, isolated baseline. It does not replace the informed SA or any TT
results in `data/sa_cluster_v2/results/`. No optimization is submitted by creating
these files. The same 240 frozen inputs and their 400 initial feasible samples
are reused: assignment and spatial Set Partitioning, n = 10, 12, 14, 40 instances
per family/size. Both quadratic and 32-scenario entropic objectives are included.

## Objective, proposal and penalty calibration

Optim.jl SimulatedAnnealing minimizes

    F(x) = f(x) + alpha * sum(abs2, A*x - b),   x in {0,1}^d.

Every binary vector can be visited. The proposal selects one coordinate uniformly
and flips that bit. It is symmetric and connects the whole binary cube; it uses
no assignment swaps, tile catalogue, feasibility repair, or learned feasible
moves. The method has constraint information through the scalar residual penalty,
which is more informative than a feasibility-only oracle. This is a different
information model from both informed SA and data-initialized symmetric TT.

Let R be the range and sigma the standard deviation of the 400 initial objective
values. Define

    scale = max(R, sigma, 1e-8 * max(1, maximum(abs, initial_costs)))
    alpha = 10 * scale.

This rule uses only the supplied initial objective values, not objective
coefficients, test outcomes, TT results, or a global optimum. The multiplier 10
is fixed for all families, sizes, objectives, and budgets. The unit is objective
value per unit squared residual; the integer constraints in these frozen inputs
ensure every nonzero squared residual is at least one. Thus a unit violation
costs ten observed initial cost ranges. This is a deliberately strong,
scale-aware penalty, **not a proof** that every infeasible point is worse than
every feasible point, and not a claim of an optimally tuned SA baseline.

The optional read-only `check_penalty_scale.jl` checks all input/objective pairs
using each one-bit neighbor of the initial best, 32 random binary vectors, and
64 random 2--8-bit perturbations. It writes `calibration.csv` and
`calibration_summary.csv` under `data/sa_penalty_v1/`. These are diagnostic
probes, not optimizer trajectories. Their objective evaluations are not included
in the benchmark because the runner neither uses nor reads these probes.

The local input-scale check covered all 480 input/objective combinations and
116,480 infeasible probes; none beat the corresponding initial feasible best
after penalization. Median alpha ranges across n = 10, 12, 14 were approximately
2321--3367 for quadratic assignment, 19.2--20.6 for entropic assignment,
4475--5897 for quadratic Set Partitioning, and 26.0--26.7 for entropic Set
Partitioning. Full values are in `calibration_summary.csv`. This check supports
the chosen scale locally but does not establish a global exact-penalty bound.

## Temperature and restarts

A large penalty with the old feasible-cost temperature would suppress virtually
every bit flip from a feasible start. Instead let v be the median of the positive
squared column norms of A, which is the typical squared violation caused by one
bit flip from a feasible point. Each chain uses geometric wall-time cooling:

    T0   = max(sigma, alpha * v / log(2))
    Tend = max(0.001 * sigma, 1e-10 * scale).

Ignoring the objective change, a typical first infeasible proposal has acceptance
probability about one half. This permits traversal through infeasible states
early in the chain; the final temperature strongly suppresses violations.
This calibration uses the penalty scale, not constraint-preserving moves.

As in the original comparison, ten equally timed chains start at the ten best
distinct initial points. Budgets are 10, 50, 200 seconds in separate runs with
the same fixed per-instance seed convention. They are not prefixes or independent
seeds. One Julia thread and one BLAS thread run per process. No objective caching
or incremental cost updates are supplied. All f(x) calls, including those on
infeasible states and repeated states, are counted. Computing A*x-b and penalty
parameters is timed. Loading, pre-run input checks, and JIT warmup are excluded.

## Comparable outcomes and diagnostics

The primary result is the **best feasible raw f(x)** seen before the deadline,
initialized by the same feasible sample as the other methods. Never compare the
penalized F(x) directly to raw TT or informed-SA objective values. The raw incumbent
history and call history have the same schema as the existing comparison.

Additional saved fields include alpha, both temperatures, counts of feasible
and infeasible objective calls, residual statistics, minimum penalized infeasible
cost, number of infeasible calls beating the initial feasible best under F,
best penalized value and its violation, and exact call checkpoints every 10,000
calls and at chain ends. These reveal an insufficient penalty or poor return to
feasibility without hiding unsuccessful runs. Repeated calls are not unique
states. Final validation calls outside the optimizer counters are excluded,
as in the earlier benchmark. Cooperative deadline overruns and real elapsed
times are saved; post-deadline improvements receive no credit.

## Cluster launch and collection

From the repository root:

```bash
bash scripts/bash/run_SA_penalty.sh
```

The wrapper installs dependencies once before submitting a four-element Slurm
array, with at most four simultaneous allocations. Each allocation requests one
exclusive c32m384 node, 32 tasks / one CPU per task, and 380000 MiB memory, with
core binding and no SMT requested. The four allocations correspond to the two
families times two objectives. No optimization runs on the login node.

There are 1,440 runs in total, with 128 private worker dictionaries saved in
`data/sa_penalty_v1/results/`. Nominal optimization is 34.7 CPU-hours in total;
the longest worker has four input/objective pairs times 260 seconds, or **17 min
20 s**. Compilation, input checks, checkpointing, allocation overhead and queue
time are additional; this is not a measured wall-time prediction.

Rerunning the same command resumes matching checkpoints. Each worker has its own
atomic file and directory lock; no shared mutable result dictionary is used.
Remove a stale `.jld2.lock` directory only after verifying the worker has ended.
Different code/inputs/settings require a new output directory, supplied with
`--output=...`; the other optional flags are `--count=...` and `--budgets=...`.

Once all workers finish:

```bash
julia --project=. scripts/sa_cluster/collect_penalty.jl
```

The collector rejects missing, incomplete, incompatible or duplicate shards,
and exports `combined.jld2`, `summary.csv`, `curves.csv`, and
`call_checkpoints.csv`. Expect `Collected 1440 penalty-SA runs.` Results can be
tracked by Git; runtime logs, locks and pending checkpoints are ignored.

## Local validation

```powershell
julia --startup-file=no --compiled-modules=existing --project=scripts/sa_cluster --threads=1,0 scripts/sa_cluster/test_penalty.jl
```

Tests cover an exhaustively enumerable toy penalty, generic bit flips,
temperature calibration, feasible incumbent bookkeeping, counts and deadline
filtering, private checkpoint dispatch/resume, and collector completeness. Full
input dispatch tests use mocked optimizers. They do not start production runs.

This baseline can reveal how much informed feasible moves contribute to SA's
advantage. It is not an upper bound on penalty-SA performance: other penalty
weights, cooling schedules, multi-bit kernels, or constraint repair could change
the outcome and would constitute separate configurations.
