# Reproducibility audit of `solve`

## Result

`solve` now has a fixed default seed and accepts an explicit `seed` keyword:

```julia
sampler! = (rng, X) -> fill_assignment_vectors!(rng, X, n)
result = solve(problem, sampler!, params, "best_cost"; seed=20260917)
```

For a fixed project environment and deterministic user-supplied cost/sampler,
repeating this call produces the same `SolverStatistics`. Serial and parallel
MPS sampling also produce exactly the same sample matrix for the same seed.

Use the same seed for the same `(n, instance)` across strategies to obtain
common random numbers in paired comparisons. Change the seed deliberately when
independent algorithmic repetitions are required, and save that seed with the
result file.

## Sources of randomness found

The complete `src/` directory was searched. The code reachable from `solve`
contained the following sources:

1. `sample_nondeg_parallel!` and `sample_nondeg!` selected MPS states with
   task-local RNGs. In the parallel implementation, the random stream depended
   on which thread happened to process a sample.
2. The feasible-sample callback called functions such as
   `fill_assignment_vectors!`, `fill_assignment_vectors_parallel!`, or
   `sample_feasible_lp!`, all of which previously used implicit default RNGs.
3. The parallel assignment sampler used implicit RNGs inside `@threads`, making
   its output sensitive to scheduling.
4. `sample_feasible_lp` and `sample_feasible_lp!` randomized their objective,
   while SCIP also has its own plugin, permutation, and LP random seeds.
5. General helpers `random_bilinear_form`, `random_assignment_vector`, and
   `generate_random_matrix_vector` used implicit RNGs. They are not called by
   `solve` when objectives are cached, but were made controllable for consistent
   experiment generation.
6. Iteration over `Set`/`Dict` values and unstable selection among tied costs
   were not random-number calls, but could change floating-point operation or
   block-construction order between processes.

`src/future.jl` also contains implicit sampling calls. That file is not included
by `src/OhMyU1.jl`, so none of those functions can be reached from the current
`solve` implementation. It was audited but intentionally left outside the
active API change.

## Changes

### Dedicated RNG streams in `solve`

`solve(...; seed=0)` creates two independent streams derived from the supplied
seed: one for feasible-data generation and one for MPS sampling. This prevents
a change in the number of random feasible points from silently shifting the MPS
sampling stream.

An RNG-aware callback with signature `(rng, destination)` is preferred. Existing
one-argument callbacks still work: immediately before each call, `solve` seeds
the current task's default RNG from its dedicated feasible-data stream.

### Schedule-independent parallel sampling

Before entering `@threads`, the code generates one seed per output column in a
fixed order. Each column then uses its own `Xoshiro` instance. Therefore thread
scheduling and thread count cannot reassign random streams to samples.

The serial implementation uses the identical per-column scheme, which makes
the `parallel=true` and `parallel=false` results equal for the same seed. The
previous zero-based `tid - 1` indexing of Julia's one-based thread IDs was also
removed.

### RNG-aware helper API

All active random helpers now have an explicit-RNG method, for example:

```julia
fill_assignment_vectors!(rng, X, n)
fill_assignment_vectors_parallel!(rng, X, n)
sample_feasible_lp!(rng, A, b, X, count)
random_bilinear_form(rng, N, density, radius)
```

The old signatures remain available and delegate to `Random.default_rng()`.
Assignment columns are now cleared before being filled, avoiding stale ones when
a buffer is reused.

### SCIP

The LP samplers now set SCIP's `randomization/randomseedshift`,
`randomization/permutationseed`, and `randomization/lpseed` from the explicit
Julia RNG. SCIP is restricted to one thread and deterministic parallel mode.

### Deterministic traversal and ties

Charge/block keys are traversed in canonical lexicographic order where their
order affects MPS construction, SVD layout, or floating-point reductions.
Candidate selection uses stable sorting, with original column order as the tie
breaker. Best and worst subsets are consequently deterministic and disjoint.

## Verification performed

The tests in `test/runtests.jl` cover:

- repeated parallel assignment generation with the same seed;
- repeated sparse bilinear-form generation;
- exact equality of serial and four-thread MPS samples;
- equality of every `SolverStatistics` field across two complete `solve` runs;
- equality of complete serial and parallel `solve` results;
- a direct repeated check of the seeded SCIP LP sampler.

## Boundaries of the guarantee

- A custom callback that creates a `RandomDevice`, constructs an unseeded RNG,
  uses external randomness, or mutates shared state cannot be controlled by
  `solve`. Use the two-argument callback and the supplied RNG.
- A wall-clock `time_limit` is intrinsically sensitive to machine load. SCIP is
  seeded and single-threaded, but if it actually reaches that limit, strict
  repeatability is not guaranteed. Prefer a deterministic node/iteration limit
  for experiments where the limit is expected to bind.
- Bitwise reproducibility across different Julia, BLAS/LAPACK, SCIP, CPU, or
  package versions is not promised. Preserve `Project.toml`, `Manifest.toml`,
  Julia version, and environment metadata with published experiments.
- The cached objective matrices/vectors are inputs rather than randomness inside
  `solve`; their exact files or checksums still need to be recorded.

