LIBRARY REGRESSION AND PERFORMANCE TESTS
========================================
Run from the repository root, with the project's dependencies installed:

    julia --startup-file=no --project=. test/runtests.jl

Native Windows PowerShell (no WSL required):

    Set-Location 'D:\Releases\OhMyU1_public'
    julia --startup-file=no --project=. .\test\runtests.jl

No cluster, external dataset, Python, downloaded PROTES, or plotting output is
needed. The orchestrator starts isolated Julia workers with 1 and 4 default
threads, BLAS=1, plus one idle interactive thread. A third tiny worker checks
--threads=4,0. Do not set JULIA_NUM_THREADS yourself; the runner controls it.

Before adding the large benchmark, the repaired suite passed 1,693 assertions
in 175.1 seconds on Windows,
including 38 seconds of package recompilation after source edits, with
Julia 1.12.5, Intel Core i5-10210U (8 logical CPUs). First package installation
or precompilation can take longer and is not a regression measurement.
The six initially exposed defects have been repaired; see KNOWN_ISSUES.txt
for the regression history. Failures remain ordinary failing tests, not
@test_broken, and do not prevent the remaining groups from running.

Coverage
--------
1. structure.jl: every amplitude, every stored block index, every link charge,
   observed-charge closure, duplicate/reordered examples, variable order,
   several saved signed random matrices, redundant equalities, a unique
   solution, one/multiple/all zero columns, reshaping round trips.
   The encoded support is the compatible charge-path closure of the training
   set. It need not equal the list of training vectors: unseen feasible
   recombinations are intentionally allowed. All candidate binary vectors
   are enumerated, including infeasible ones, and compared element by element.
2. sampling.jl: train first, then verify exact conditional probabilities and
   all 512 serial/parallel draws per fixture. Serial seeds must replay exactly;
   parallel draws need not equal serial draws or replay across schedules.
   Samples must stay in support and satisfy Ax=b. Sampling must not change
   tensor amplitudes. An additional 16,384-draw categorical check compares actual
   frequencies with the Born probabilities using a conservative eight-standard-
   deviation envelope; the exact conditional tests are not statistical. Both one- and four-thread configurations are exercised.
   The no-interactive-pool worker also checks empty/short batches, untouched
   trailing output columns, invalid buffers and eight concurrent nested calls.
3. exact_sampling.jl: the actual assignment constructor in
   scripts/protes/assignment_memory.jl for n=1,2,3 (all binary vectors), and
   exact set partitioning. The latter uses independent forward/backward
   subset-sum charges fed into the library initializer; the library does not
   currently expose a full set-partitioning MPS constructor. Production
   locality_ranks and ExactPartitionSampler are checked against exhaustive
   enumeration, explicit unfolding ranks and exact rational path probabilities.
   Source RNG replay, views and concurrent independent generators are checked.
4. gradients.jl: nonuniform signed real cores, individual core components,
   both flattened forms, and every merged component. The reference loss is
   computed by contracting independent dense tensors over the whole cube.
   Dense external environments form an independent Jacobian. Central finite
   differences are compared component by component (h=1e-6, absolute tolerance
   3e-7, relative tolerance 3e-6). Full right/left production sweeps, including
   SVD projection, are compared against sweeps driven by finite differences.
   Tests include zero columns, non-unit target weight sums, duplicate examples
   with split weights, scale invariance, and learning the explicit [0.8,0.2]
   categorical target. This covers the real sector-dimension-one training API;
   it does not claim coverage of unfinished generic/complex implementations.
5. selection.jl: exact selected column identities for BestCost and all three
   article ADD strategies (mixed, weighted rank, best nonzero mixed), weights
   0,0.1,0.2,0.3,0.4,1, barriers, graph distances, all-known fallback and input
   immutability. Ties and selection diagnostics have separate contract tests.
   BestCost is checked with a graph callback that throws if called. Already
   evaluated optima discarded by ADD still update the incumbent without extra
   objective evaluations. Zero-temperature and singleton training are covered.
   The GDF policy may return fewer points if too few positive-distance points
   remain; tests preserve this existing policy rather than inventing padding.
6. structure.jl: amplitude-preserving orthogonalization, block isometries,
   full-enumeration norm, normalization and idempotence.
7. EELS uses the saved cardinality fixture: sum(x)=2, training 1100 and 0011.
   Residual charge 1 at cut 2 is hidden. Every tested dataset ordering must
   recover it, increase support above two points, preserve old points and
   never admit an infeasible point. No random chance is needed for this test.
8. performance.jl: real solve(), assignment 5x5 (25 binary variables), 32
   initial points, 512 model draws, three optimization iterations, both serial
   and parallel sampling. The initial set excludes the identity optimum;
   subsequent external batches include it. The final exact cost must be zero,
   the incumbent feasible, and the learning curve monotone. Each mode is warmed
   once, then measured three times. Times are medians, excluding compilation
   and the explicit GC before each measurement. End-to-end allocations are
   total Julia allocations per run, NOT peak live memory. Process peak RSS is
   reported and checked separately. The benchmark runs FIRST in both workers,
   so this RSS includes the same imports and benchmark JIT, not different
   preceding correctness suites. It is still a process peak, not tensor size.

Performance check after the sampler repair
-----------------------------------------
A direct old/new comparison used the same assignment 8x8 MPS built from 400
feasible examples, 5,000 draws per batch, BLAS=1, warmed calls, alternating
measurement order and explicit GC outside each timed region. On four default
threads plus one interactive thread (9 repetitions per version), old/new median
sampling times were 80.1/73.3 ms; allocations were both about 48.0 MB. On one
plus one thread, a noisy initial nine-repeat run suggested a 12.3 percent
slowdown. A follow-up with 25 repetitions and matched per-pair seeds measured
251.6/257.7 ms (2.4 percent slower). These are local timing observations, not
speed guarantees; the first noisy result is not evidence of a stable regression.

The complete solver test measured 35.8/37.8 ms for serial/parallel sampling
on one default thread and 36.8/25.0 ms on four. All historical runtime,
allocation and peak-RSS guards passed without changing the stored reference.

Performance policy
------------------
The checked-in fixtures/performance_reference.toml contains the observed
performance on the machine above. It is explicitly a performance reference,
not a declaration that functional tests pass. Same-machine historical checks
require the exact CPU model, logical CPU count, platform and Julia version.
Algorithm source hashes are deliberately NOT part of this match: their
changes are precisely what the regression tests should detect.

Per mode/thread count, a regression fails if time exceeds 1.5 times reference
plus 20 ms, or allocations exceed 1.25 times reference plus 1 MB. Peak RSS may
increase by at most 25 percent plus 64 MB. The time floor accounts for the short
workload and OS scheduling noise; this is a coarse guard, not a microbenchmark.
Run with other CPU-intensive applications closed and consistent power settings.

Portable ceilings: 8 seconds and 350 MB allocated per warmed solve, process
peak RSS below 2.5 GB. Across thread counts, 4-thread runtime may not exceed
max(100 ms, 2.5 times 1-thread runtime); allocations may not exceed twice the
1-thread amount plus 20 MB; RSS may not exceed 1.5 times plus 128 MB. Four threads
are not required to speed up a small task. A foreign checked-in CPU baseline
is reported as inapplicable, never treated as a valid historical comparison.

After all correctness issues are fixed, explicitly establish a local baseline:

    julia --startup-file=no --project=. test/runtests.jl --calibrate

Calibration refuses to bless a failing suite and never updates the checked-in
reference. test/local_performance_baseline.toml is ignored by Git. A stale local
baseline from another CPU/Julia must be removed or explicitly recalibrated;
measurements are never silently adopted as a new baseline.

Reports and maintenance
-----------------------
Logs, timing/memory TOML files and summary.toml are in test/output/ (ignored).
The tests hash src/*.jl before and after running, and never edit algorithms.
The readable fixture matrices are checked in; RNG changes cannot change them.
Change the workload identifier if the performance fixture/parameters change.
The full non-threaded correctness suite runs once; sampling and selection are
repeated with four threads. The default run also includes the large assignment
benchmark described below and intentionally exceeds the original three-minute
budget. It is not silently skipped.
There is no whole-machine runtime promise on arbitrary hardware or a cold cache.

Large assignment benchmark (enabled by default)
----------------------------------------------
large_performance.jl runs the public solve() API on assignment n=10: 100 binary
variables and 20 equality rows (one redundant), a fixed random linear objective,
1,000 initial feasible examples, 5,000 MPS samples per iteration, 10 global
iterations, two training sweeps per iteration, learning rate 0.05, 5 percent
elites, no worst-point retention, BestCost selection, no EELS and no barrier.
The library fills its 20,000-candidate pool using fresh uniform permutations
alongside unique MPS samples and elites. The external sample count is therefore
variable; it is not another fixed batch of 5,000. No optimum is injected.

Seeds: objective 8101, independent external-source RNG 8102, model/task RNG 8103.
Source RNG consumption is independent of model RNG calls; the number requested
by the solver still depends on model uniqueness. One worker uses one default
thread and serial sampling; another uses four default threads and parallel
sampling. Each has one idle interactive thread and BLAS=1. They run sequentially
in isolated processes to avoid interference and comparable peak-RSS accounting.

One iteration with the same dimensions warms each worker before timing. Then
three complete solves are measured; the report retains every run and the median,
GC time, initial/final costs, cumulative allocated bytes, process peak RSS,
load time and warm-up time. Explicit GC before a run is outside the timer;
GC occurring during the solve is included. Cumulative allocation over all ten
iterations is NOT peak live memory. RSS includes imports, JIT and warm-up as
well as optimization, and cannot be interpreted as the MPS storage alone.

Assertions check feasibility, binary output, consistent final cost, monotone
learning curves, source-call counts and a strict improvement over the initial
set. They do not require the two random optimization trajectories to coincide.
Portable ceilings are 120 seconds per median solve, 15 GB cumulative allocation,
and 3 GB peak RSS. Four-thread time may not exceed 1.5 times one-thread time;
allocations may not exceed 1.25 times plus 50 MB; RSS may not exceed 1.5 times
plus 128 MB. A matching hardware/Julia/workload historical reference additionally
limits time to 1.3 times reference plus one second, allocations to 1.2 times plus
50 MB, and RSS to 1.25 times plus 64 MB. These thresholds are fixed before the
initial reference is recorded. They detect substantial regressions, not small
OS scheduling fluctuations. Foreign checked-in references are inapplicable.

Run ONLY this benchmark from native PowerShell:

    julia --startup-file=no --project=. .\test\runtests.jl --large-only

The ordinary command runs everything. --calibrate requires the full suite to
pass and records separate ignored local references for small and large workloads;
it cannot be combined with --large-only. The original small reference remains
unchanged. Detailed large reports are test/output/large_threads_1.toml and
large_threads_4.toml. summary.toml separates the quick-stage and total wall time;
the ordinary workers also record import time and timings for each test file.

Initial large reference (2026-10-12, same Windows/Julia/CPU as above)
----------------------------------------------------------------
One thread, serial sampling: runs 29.94/29.07/28.39 s, median 29.07 s,
4.433 GB cumulative allocation per solve, 1.224 GB process peak RSS.
Four threads, parallel sampling: runs 22.10/21.03/20.81 s, median 21.03 s,
4.480 GB cumulative allocation per solve, 1.194 GB process peak RSS.
All six runs improved cost from -11.3536352181 to -13.9418709245.
GB here means 1,000,000,000 bytes, not GiB. Four-thread elapsed time was
27.7 percent lower in this measurement; this is not a universal speedup claim.

All 1,749 assertions passed. The entire default run took about 7 minutes
19 seconds, including imports, JIT, warm-ups, explicit GC, repeated measurements
and correctness checks. The short stage alone took 185.2 s in this run, showing
that whole-process elapsed time varies appreciably on this machine. The earlier
175.1 s run included 38 s of package precompilation; its difference from 130.8 s
must not be interpreted as the optimizer slowing down by the same proportion.
The benchmark's timing arrays isolate warmed optimization from those costs.

Focused InvXindex audit
----------------------
index_lookup.jl verifies the mapping against physical charges computed directly
from A and the variable domain. Distinct block entries exercise both matrix-valued
getindex overloads and all four scalar getval overloads, including missing blocks,
zero columns, saved signed problems and a three-value physical domain. Matrix
accessors are also checked with 2x3 bond dimensions. This is an accessor test,
not a claim that generic nonbinary training/sampling is supported. Its 1,127
assertions passed in a focused run; it is included in the ordinary one-thread
correctness worker. Algorithm implementation files were unchanged by this audit.

Cached training positions
-------------------------
collect_left/right now retain a (bond charge, physical value) -> first flattened
coordinate lookup in U1Matrix.Xpositions. For nondegenerate bonds this is the
entire row/column index. XboundsDict keeps its original split-boundary meaning.
The lookup is built from the actual concatenation order; absent sectors do not
create artificial rows, and a zero column preserves each physical state.
index_lookup.jl additionally checks cached indices against an independent sector
search, reversed block insertion order and missing-sector errors. The reference
search remains in tests only. train_nondeg! computes the target weight sum once
and passes it through all sweeps; direct sweep calls compute it once per call.

After caching training positions, all 3,496 assertions passed in 326.1 seconds.
Large-workload medians were 16.84 s (one thread) and 12.22 s (four threads), with
4.506/4.555 GB cumulative allocation. The approximately 1.7 percent allocation
increase is the cost of the position cache. Historical runtime, allocation and
RSS guards passed without recalibration. Different machine load means the
observed timing improvement is not an isolated causal estimate of this change.

Fast paths and reference backends (2026-10-12)
---------------------------------------------
TrainParams(rate, rank) and TrainParams(rate, rank, cutoff) remain valid. The
default backend=:fast uses scaled dominant singular triplets for 1xn, nx1 and
2x2 real blocks; larger blocks still use LAPACK. Choose
TrainParams(rate, rank; backend=:lapack) to use library SVD for every block.
orthogonalize!(mps; backend=:lapack) restores the library LQ path. The default
canonicalizer directly normalizes one-row sectors when both bond degeneracies
are one; degenerate bonds retain LQ. solve(...; factorization_backend=:lapack)
selects both library factorization paths without changing SolverParams files.
At repeated leading singular values, the best rank-one approximation need not
be unique; tests compare singular values and approximation errors, not signs.

Training target weights are aggregated by (middle charge, left physical value,
right physical value) once per train call. The merged derivative contribution
is 2*learning_rate*sum_weight/(left_value*right_value). The two core values are
read from the already reshaped matrices. No gradient term is dropped. Target
charges are also computed once and shared by all sweeps. TrainParams floating
fields have concrete types to avoid dynamic dispatch in arithmetic.

sample_nondeg! and sample_nondeg_parallel! now build integer transition/CDF
tables for each batch, then draw without charge hashing in the inner loop.
Tables require right-canonical, nondegenerate cores and are not retained across
model updates. Pass backend=:legacy to either sampler for the former path.
The new path supports unequal finite physical domains and zero columns. RNG
ownership remains task-local. Floating-point regrouping and factorization
gauges can change exact random trajectories; reproducibility means replaying
the same implementation and runtime settings, not bitwise equivalence between
old and new implementations.

Graph-distance batches use task-owned scratch, with no per-point Channel
operations. All constraint matrices and residual charges use the same general
vector-based implementation; there is no binary-charge specialization or
machine-word constraint-count limit. User-provided distance callbacks are preserved.

fast_paths.jl independently checks extreme-scale Float32/Float64 triplets,
zero and repeated singular values, direct canonicalization, zero rows, complete
training with both backends, split target weights, exact dense probabilities,
nonbinary physical domains, signed charges, large constraint counts, and nested
concurrent calls. Existing finite-difference, strategy,
support and sampling tests remain enabled. Historical performance thresholds
have not been relaxed or recalibrated.

Additional warmed BestCost/GDF profiling, independently of the test suite:
    julia --startup-file=no --project=. --threads=1,1 test/benchmark_assignment.jl
    julia --startup-file=no --project=. --threads=4,1 test/benchmark_assignment.jl
Run these sequentially. Each measures three complete 10-iteration solves per
strategy on assignment n=10 after warm-up. GDF uses weight 0.4; other settings
match the large regression workload. Optional first argument sets the TOML
output path. Source hashes, machine details, all timings, cumulative allocation,
costs and process peak RSS are saved, followed by a separate sampling profile.
Profile overhead is excluded from the reported timings and RSS measurement.
See OPTIMIZATION.txt for the measurement history and current follow-up results.
fixtures/optimization_20261012.json retains the original before/after measurements
from the revision that still included the subsequently removed binary masks.
