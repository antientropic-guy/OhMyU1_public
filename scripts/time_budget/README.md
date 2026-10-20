# Equal elapsed-time comparisons

From the repository root on the cluster:

```bash
bash scripts/bash/run_time_budget.sh
```

This submits **one Slurm array with four allocations**, each reserving 32 physical
CPU cores and 380000 MiB on `c32m384`. Each allocation runs 32 isolated Julia
processes. Each solver uses one CPU, one Julia thread, one GC thread, and one
BLAS thread. The paired arms run sequentially on the same pinned CPU; instance
parity reverses arm order. There is no shared mutable result dictionary.
The script performs no optimization on the login node. Package preparation and
input checks occur once before submission; workers never call `Pkg.activate`.

## Frozen comparisons

| Comparison | Inputs | Pairs |
|---|---|---:|
| ADD GDF (weight 0.4) vs Best Cost | Assignment test linear and quadratic; n=12,...,16; 50 instances each | 500 |
| Persistent EELS (gamma=1) vs persistent Best Cost | The same assignment test objectives | 500 |
| ADD GDF (weight 0.4) vs Best Cost transfer | Existing locality grid, entropic objective; spans 5,6,8,10,12,15,18; m=18,19,20; n=13,14,15; 15 instances per combination | 945 |

Total: **1945 pairs / 3890 arms**. All seven locality groups are included:
135 instances per group. Inputs and original results are never modified.
Here `n` is the size parameter; the binary vector has `n^2` components.
For the locality grid, `m` is the number of equality constraints.
The transfer grid has already been used in the earlier transfer analysis; this
is a time-budget reanalysis, not an independent new test sample.

Each arm runs once up to **200 seconds**, with checkpoints at **10, 50 and 200
seconds**. There is no 20-iteration stopping limit and no added stagnation
restart. Defaults can be changed:

```bash
bash scripts/bash/run_time_budget.sh --budgets=20,100,400 --output=data/time_budget_400s
```

The ideal compute-only makespan at the default maximum budget is
3890*200/128 = 6078 seconds (1.69 hours). Startup/JIT warm-up, source preparation,
saving, imbalance, and safe-point overruns increase this. Allow roughly 2–4
hours as a planning estimate, not a measured cluster runtime. Slurm permits
24 hours. Competing jobs belonging to the user must still respect the user's
four-allocation quota.

## Algorithm parameters

* Assignment ADD: 1000 initial feasible draws, 10000 TT draws per iteration,
  candidate pool filled to 20000 with fresh uniform random permutations,
  selection target 1000, 5% retained elites, GDF weight 0.4.
  Both strategies rebuild the TT each iteration. Sampling uses `table`.
  The initial dataset and per-iteration source RNG prefixes are paired;
  requested external batch lengths may differ.
* Transfer ADD: the saved 500 initial draws; 5000 TT and exactly 5000 fresh
  exact-source draws per iteration; selection target 500; 5% retained elites.
  Both strategies rebuild the TT each iteration. Sampling uses `table`.
  Source RNGs follow the existing transfer protocol.
* EELS: 400 initial draws; 10000 TT draws per iteration; train on at most the
  best 400 unique current draws. No external data after initialization; no
  accumulation of the training set; one model initialization; learned
  amplitudes persist. Only EELS expands the charge graph, once, with gamma=1.
  Sampling uses `legacy`, matching the existing persistent EELS comparison.
* All: one training sweep, learning rate 0.05, charge degeneracy one,
  no barrier or worst retention, existing temperature rules unchanged.

The original ADD and locality functions are cloned in memory with deadline
hooks. The persistent EELS function is reused through its observer.
Production library files are unchanged. Analysis-only external graph-distance
diagnostics in the old transfer driver are omitted from **both** arms; actual
GDF selection is unchanged. Tests compare final costs with the original
functions on small nonconstant examples for both arms and all objective types.

The assignment ADD protocol has no objective cache. Transfer and EELS retain
their existing objective caches in both arms. Counts always mean actual calls
to the uncached objective, not cache lookups. Do not pool call-count comparisons
across these different protocols.

## Timing, fairness, and deadlines

Compilation/warm-up, loading frozen inputs, preparing the common initial
dataset and the exact external sampler are outside the clock and are shared
input preparation. Initial objective evaluation, TT construction, training,
sampling, subsequent external draws, selection, caching, and instrumentation
are inside the clock. The initial samples are identical within every pair.
Every objective improvement is timestamped on completion. Only improvements
completed before a checkpoint count; the last iteration need not have finished.
Objective-call counts at the three budgets count completed actual evaluations.

The timer is cooperative: the current linear-algebra, training, or sampling
operation can finish after 200 seconds before a safe-point check stops it.
**Late improvements are discarded.** Actual elapsed time and overshoot are
saved explicitly. This preserves valid capped quality comparisons, but a
large overshoot must be disclosed in a runtime comparison. Timing is wall time,
not CPU time. Parallel workers can still share memory bandwidth and frequency
limits; both paired arms use the same allocation.

## Results, memory, and resumption

Output: `data/time_budget_v1/`. The manifest freezes code hashes, input hashes,
task list, and budgets. Each task/arm owns `result.jld2`, plus portable CSVs:

* `improvements.csv`: every improved incumbent, time, objective calls, iteration.
* `iterations.csv`: completed iteration checkpoints.
* `objective_calls.csv`: approximately 0.1-second call-count observations at
  solver safe points. This trace can be sparse during long training kernels;
  exact requested-budget counts are in `checkpoints.csv`.
* `checkpoints.csv`: quality and completed objective calls at each budget.
* `result.jld2`: configuration, initial-data hash, task identity, full recorded
  diagnostics, final incumbent, `res_dict`, allocation volume, GC time, and
  exact structural ranks for transfer instances. `res_dict` is an arm/n/instance
  dictionary; its learning curve lists incumbent improvements, not iterations.

An external Python monitor samples Linux `/proc` every second. It records RSS,
anonymous RSS, process high-water RSS and CPU seconds in `monitor/`.
Worker processes are reused; use each arm's start/end timestamps to select
its RSS samples. Process high-water RSS includes earlier arms and warm-up,
so it must **not** be interpreted as an arm-specific peak.
RSS includes the runtime and retained Julia heap pages. Sampling can miss
brief peaks. Allocation volume is cumulative, not live memory.

After completion (or while inspecting partial results):

```bash
julia --project=. --threads=1 scripts/time_budget/collect.jl
```

This creates `summary.csv` and `missing.txt`. Memory/allocation/actual-time
columns describe the whole arm, repeated for its three quality checkpoints.
The original traces remain available for more detailed plots.

Rerun the same Bash command to resume. Completed arms are validated and skipped;
an interrupted unsaved arm restarts. Writes use an atomic final rename.
An atomic per-pair `.running` directory prevents duplicate writers. If Slurm
kills a worker, remove that directory only after confirming the old allocation
has stopped. Changing code, input files or budgets requires a new output
directory; incompatible measurements are never silently mixed.

Local checks (no submission):

```powershell
julia --project=. --threads=1 --gcthreads=1 scripts/time_budget/test.jl
julia --project=. --threads=1 --gcthreads=1 scripts/time_budget/run.jl --smoke --budgets=0.2,0.5 --output=tmp/time_budget_smoke
```
