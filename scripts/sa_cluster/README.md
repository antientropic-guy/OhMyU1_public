# Isolated SA / MPS cluster benchmark

This experiment extends the original `SA_learning_diagnostics.ipynb` protocol.
The older pilots are ignored by Git and are not dependencies of this directory.
No cluster submission is made by preparing the data or running the toy tests.

## Frozen design

Four Slurm allocations cover the Cartesian product of family (assignment / spatial
strip Set Partitioning) and objective (quadratic / entropic). Each allocation
requests an exclusive node, 32 single-core tasks, 380000 MiB RAM (about 371.1 GiB)
and 48 hours, and launches 32 independent
single-core Julia processes with one `srun`. At most four allocations run at once.
Each process has its own result file; methods are sequential within a process.

Sizes are n = 10, 12, 14, with n squared binary variables and 40 instances per
(family,n), selected by IDs 1--40 before optimization. Both objectives, three
methods and independent 10, 50, 200 second budgets give 4,320 optimizer runs.
Total nominal optimization is 240 * 2 * 3 * 260 = 374,400 seconds = 104 CPU-hours.
Each allocation processes 120 instance/objective pairs: 24 processes get four
pairs and eight get three. Each pair costs 3 * 260 = 780 nominal seconds, so
the longest process costs 3,120 seconds = 52 minutes (ideal perfect balancing:
48.75 minutes). This excludes compilation, structural checks, saving and cooperative
deadline overruns. Queue time, memory bandwidth contention and preparation make
actual elapsed time longer; no cluster measurement is available yet. Memory is
shared across the allocation (about 11.6 GiB per active process on average, not an enforced
per-process limit); simultaneous full n=14 models can increase memory pressure.
Other jobs already owned by the user can delay scheduling.

The reported c32m384 node capacity is 64 Slurm CPUs and 380000 MiB. The CPU count
alone does not distinguish physical cores from SMT threads. We retain 32 processes
with `--hint=nomultithread` and core binding rather than assuming 64 physical cores.
Exclusive allocation avoids competition with unrelated jobs on sibling threads.
The launcher logs `scontrol show job` and `scontrol show node` for actual topology.
The former request of 384G meant 393216 MiB, exceeding the reported node capacity.

| Family | n | Binary variables | Equalities | Mean maximum exact rank ± SD | Mean of mean exact ranks ± SD |
|---|---:|---:|---:|---:|---:|
| Assignment | 10 | 100 | 20 | 461 ± 0 | 175.86 ± 0 |
| Assignment | 12 | 144 | 24 | 1715 ± 0 | 601.52 ± 0 |
| Assignment | 14 | 196 | 28 | 6434 ± 0 | 2100.52 ± 0 |
| Strip Set Partitioning | 10 | 100 | 30 | 27.70 ± 5.16 | 12.37 ± 1.04 |
| Strip Set Partitioning | 12 | 144 | 36 | 36.05 ± 5.34 | 16.16 ± 1.08 |
| Strip Set Partitioning | 14 | 196 | 42 | 45.35 ± 5.52 | 19.83 ± 1.39 |

Ranks are total bond dimensions of the full exact indicator in the specified
binary variable order, not dimensions of degenerate charge blocks. They are
computed exactly by coverage-state dynamic programming / enumeration, without
estimating them from the initial sample. SP maximum ranks range from 20 to 39
at n=10, 27 to 49 at n=12 and 37 to 59 at n=14. No instances were filtered by optimization outcomes.
The full assignment n=14 model is not truncated to the rank of its sampled model.

## Inputs and objectives

`data/sa_cluster_v2/inputs/` contains 240 standalone compressed JLD2 records.
Copy this entire directory and `manifest.jld2`
to the cluster with the repository. The runner does not need the older SA pilots
or the source input directories. Each record includes A, b, initial points, both
objectives, seeds, tile definitions where applicable, full rank profile, exact
support count and provenance hashes. `rank_summary.csv` summarizes the snapshot.

SP uses the first 40 existing `light` strip test instances at n=12,14. The n=10
instances use the same generator and seed convention (811000000 + 1000n + ID).
`extend_n10.jl` creates the isolated v2 snapshot, copying all 160 old input files
byte-for-byte and generating 80 new records. It refuses to overwrite v2.
The n=10 quadratic matrices use independent seeds 912010000 + ID, with the same
symmetrized uniform [-7,7] construction as the older test generator. Their raw
sources are saved under `sources_n10/`; none of the old source files are changed.
Tiles are
ordered by minimum geometric cell, with the singleton last for that cell: the
same fixed low-rank spatial order as in the prior strip pilot. There is no variable
order search. Rows are restored to geometric cell order. Sampling uses the exact
BigInt completion-count generator; every task receives 400 independent uniform
feasible points with replacement. Assignment uses independent uniform random
permutations, encoded in the original row-major binary order. The initial dataset
for a task is identical for every method, objective and budget. It is evaluated
and ranked afresh under the selected objective.

Quadratic cost is x'Qx, using the existing test matrices `q_n_i_r=7_test.jld2` at n=12,14
and the new matrices described above at n=10,
including diagonal and both off-diagonal directions. For each (n,i), exactly the
same Q is supplied to assignment and SP in their fixed coordinate orders.

The nonpolynomial cost is the original 32-scenario entropic objective:

    f(x) = c'x / s + tau * log(mean(exp(W*x / (tau*s))))

It is evaluated with a stable log-sum-exp. The frozen c, W and tau are taken from
the corresponding SP instance and permuted with its columns; s = sqrt(3n).
Assignment gets exactly these same arrays and scale. Thus neither the coefficient
distribution nor the scale is changed between the two feasible sets. This is a
shared cost on differently constrained binary spaces; raw objective magnitudes
across the families are not a measure of relative difficulty. Objective parameters
and initial-data generator seeds are stored, not regenerated during cluster runs.

`prepare.jl` regenerates a snapshot from source files only when no manifest exists.
It cross-checks all feasible points, the original SP row/column permutation, and
exact support counts using two independent dynamic programs. It deliberately
refuses to overwrite an existing frozen snapshot. No regeneration is needed for
the supplied data.

## Methods and timing

- **SA:** Optim.jl SimulatedAnnealing, ten equally timed chains started from the
  ten best distinct initial points. Temperature goes geometrically from the
  initial cost standard deviation to 0.001 times that scale during each chain.
  Assignment proposals transpose two assigned columns. SP proposals uniformly
  replace the cover of a randomly selected tile patch; crossing tiles yield
  self-loops. Both kernels are symmetric and preserve feasibility. No cost deltas
  or objective caching are supplied.
- **Data + EELS:** rebuild from the current elite dataset and expand with gamma=1
  on every iteration; normalized uniform initialization followed by training.
- **Full MPS:** construct the full normalized indicator once inside each timed
  trial, then reset to its uniform amplitudes on every iteration.

Both MPS methods use the legacy sampler, nondegenerate charge blocks, learning
rate 0.05, one training sweep and up to 10,000 samples per iteration (100-point
batches for deadline checks). They retain the best 400 distinct points from the
previous elites plus new samples. Target weights are Boltzmann weights with the
standard deviation of current elite costs as temperature. No external feasible
data is injected. This is explicitly the old reset/rebuild protocol, not the
later persistent-model or stagnation-restart comparison.

All methods receive the same black-box objective implementation. Every method
runs sequentially on the allocated CPU with one Julia and one BLAS thread.
Method order rotates by instance ID. Each allocation has one objective. Seeds are fixed, but different
budgets are independent trials, not prefixes: SA cooling depends on the budget.
There is one optimization seed per instance, not 40 repeated seeds on one task.

Compilation warmup, input loading, input validation and SA move-catalogue
preparation are excluded from optimization time, as in the original pilot.
Initial objective evaluations and MPS construction/copying/training/sampling
are included. A sweep or construction can exceed the deadline; actual time and
zero-iteration cases are recorded, and no post-deadline improvement is credited.
In particular, 10 s can be insufficient for the full assignment n=14 model.
The 200 s budget is included to assess this without silently excluding setup cost.

## Launch and resume

From the repository root on the cluster:

```bash
bash scripts/bash/run_SA_cluster.sh
```

The wrapper instantiates the root and pinned Optim environments once on the
submission host (no optimization there), then submits array 0--3 with concurrency
limited to four, each using 32 CPUs. Julia 1.12 is required by the root project; Julia 1.12.5 / Optim
2.3.2 match the supplied environment. First-time dependency installation requires
package access. Compilation uses existing caches and ordinary in-memory JIT;
there is no inter-worker synchronization or shared results dictionary.

Each completed method is checkpointed atomically into its worker file under
`data/sa_cluster_v2/results/`. Running the same command resumes completed workers
and skips completed methods. Code, input, budget and package-version fingerprints
must match existing results. Do not edit the adapters midway through a run.
Directory locks prevent two simultaneous writers to one worker file. After a hard
kill, remove that worker's `.jld2.lock` directory only after verifying that the
corresponding Slurm job is no longer running; normal exits remove it automatically.
Changing the design requires a separate `--output=...` directory.

## Saved information and collection

There are 128 private files named `family_objective_workerK.jld2`.
Each worker stores `results[(n, instance, objective, budget, method)] => Dict` and
its full protocol. Result dictionaries contain the initial/final best costs,
best feasible solution, incumbent histories versus time and objective-call count,
actual elapsed time, completed iterations/chains, sample/proposal counts,
temperature and iteration logs for MPS, initialization and build/train/sample
times, input hash, hardware description and Slurm identifiers.

Julia allocated bytes and GC time cover the adapter call, including final checks;
they are not peak live memory. `process_maxrss_bytes` is the process high-water
mark since worker startup, not a method-specific peak, and must be interpreted
accordingly. Detailed function-call counts include duplicates, self-loops and
Optim's repeated incumbent evaluations. They are not unique-state counts or
intrinsic query-complexity bounds. Hardware can differ across workers; paired
method comparisons within a worker use the same allocated CPU.

After all 128 files have been downloaded (or on the cluster), run:

```bash
julia --project=. scripts/sa_cluster/collect.jl
```

The collector rejects incomplete or incompatible shards and writes a combined
dictionary keyed by `(family,n,instance,objective,budget,method)`, `summary.csv`,
and `curves.csv`. The raw dictionaries retain details not flattened into CSV.
Completed results in `data/sa_cluster_v2/results/`, inputs, and new scripts can be tracked by Git. Runtime logs, lock directories, and pending checkpoint files remain ignored.
The completed cluster analysis is in `examples/SA_cluster_analysis.ipynb`.
It includes 36 separate experimental groups, mean learning curves versus time
and measured objective calls, exact call totals by time budget, paired tests,
tables, timing checks, and limitations of the reset/rebuild protocol.
Rebuild it without running any optimizer:

```bash
julia --project=. scripts/sa_cluster/collect.jl
julia --project=. --threads=1,0 scripts/sa_cluster/analyze.jl
python scripts/sa_cluster/build_analysis_notebook.py --execute
```

The Python generator requires numpy, nbformat, nbclient, and ipykernel. Figures
are generated by Julia. Analysis outputs live in `examples/sa_cluster_analysis/`.
The audit checks input fingerprints, saved protocol, all final solutions and
objective values, paired initial costs, recorded histories, and CPU thread counts.

## Local validation, not a benchmark

```powershell
julia --startup-file=no --compiled-modules=existing --project=scripts/sa_cluster --threads=1,0 scripts/sa_cluster/test.jl
```

This checks toy exact indicators component by component, feasible moves,
objective formulas, a few bounded optimizer steps, and all input checksums and
paired objectives. It does not launch the 40-instance comparison or submit jobs.

Additional tests: `test_io.jl` (root project) checks synthetic shard collection
and rejects incomplete results; `test_dispatch.jl` (this Optim project, one thread)
replaces both optimizers with mocks and checks checkpointing, resume, incompatible
protocol rejection, and duplicate-writer locks. These also perform no benchmark.
