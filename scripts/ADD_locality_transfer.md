# Frozen ADD transfer experiment

Run from the repository root on the cluster:

```bash
bash scripts/bash/run_ADD_locality_transfer.sh
```

## Recovery of the interrupted activation-lock run

After stopping the old allocations and updating the checkout, resume the
`a372376ce3047947` run with its 744 complete pairs:

```bash
bash scripts/bash/run_ADD_locality_transfer.sh --resume-from a372376ce3047947 --expected-completed 744
```

This first runs a sequential checkpoint audit without optimization or Pkg
operations. It requires the old implementation hashes to match the current
implementation except for the exact removal of redundant `using Pkg` and
`Pkg.activate` lines. Numerical parameters, Julia version, frozen input hashes,
initial training fingerprints, completed histories, feasible incumbents and
their objective values are checked. A mismatch aborts before Slurm submission.
If the actual completed-file count differs, inspect it and explicitly supply
the correct count; incomplete `.tmp` files are never imported.

The original directory remains unchanged. Validated results are copied into
the current implementation's output directory, with the original file hash
and code hashes retained in `recovery_provenance`. Repeating this migration is
idempotent. The ordinary worker resume logic then skips those pairs and computes
only missing pairs (201 if exactly 744 were saved). All four allocations are
submitted with the original task assignment; fully completed workers exit after
checking their checkpoints. Old `.running_*` directories are not copied or
deleted. Final analysis must use the new output directory printed by recovery,
not combine old and new directories and count the copied results twice.

Workers now rely on `--project=.` and do not call `Pkg.activate`. Package
installation remains a single login-process operation for fresh launches only;
the recovery command skips it entirely. Invoke the Julia scripts with the project
explicitly selected, including when running the generators directly.

Run the isolated recovery regression checks locally:

```bash
julia --startup-file=no --compiled-modules=existing --project=. --threads=1,0 scripts/test_resume_ADD_locality_transfer.jl
```

The checks use synthetic solver records on one frozen input; no optimization or
production-result mutation is performed.

This submits four simultaneous exclusive c32m384 allocations, each with 32
single-threaded Julia processes on physical cores and 380000 MB memory. BLAS is
single-threaded. No optimization runs on the login node. No inputs are regenerated.
At most 128 pairs are active; the two arms within each pair run sequentially.

The input grid is `data/set_partitioning/locality_grid_v1/instances/`:
span w = 5, 6, 8, 10, 12, 15, 18; m = 18, 19, 20 equalities; n = 13, 14, 15,
with **N = n squared binary variables**, not n variables. Each of the 63
combinations has 15 frozen instances: 135 per span, 945 total. These are transfer
instances, separate from the assignment objectives used to choose the ADD weight.
They have previously been inspected in exploratory locality studies; this is
not a claim of a newly untouched or preregistered test set.

Only Best Cost and the selected **GDF, weight 0.4** (`best nonzero mixed`) are run:
1890 solves total. This is the common best observed configuration in the corrected
linear and quadratic assignment tuning. No EELS or hyperparameter search is run.
The objective is the saved 32-scenario entropic cost with tau = 0.5 and scale sqrt(m).

The previous locality setting is retained: 500 initial feasible draws, 15
iterations, one sweep at learning rate 0.05, 5000 TT draws and 5000 fresh exact
uniform feasible draws per iteration, 5% elite retention, no worst retention,
charge degeneracy one. TT is rebuilt from selected data each iteration. Sampling
uses the explicit table backend (the current public solver default).
Both arms receive identical initial data and independent,
matched external RNG streams. Their model distributions, and hence model samples,
can differ. Uniform weights handle zero cost spread. The incumbent includes all
evaluated candidates, including the final external batch. This tests transfer of
the selection rule/weight, not identity of all assignment sample budgets.

The runner reuses `locality_add_arm` in `run_PartitionLocality.jl`. Each instance
has a private atomic checkpoint and directory lock; workers never share mutable
solver state. A content hash of the implementation selects a fresh output folder:

`data/set_partitioning/locality_grid_v1/results_add_transfer_v3/<hash>/entropic/`

Files are `res_ADD_entropic_span<w>_m<m>_n<n>_i<i>.jld2`. They retain the usual
`res_dict["best_cost"][n][i]` and
`res_dict["best nonzero mixed"][0.4][n][i]` schema, plus configuration, solver
parameters, input/code hashes and paired diagnostics. Structural ranks and exact
support counts, per-iteration source hashes, graph novelty, learning curves and
unique cached objective counts are saved. These timings are concurrent diagnostics,
not isolated speed measurements. Repeat the command to resume; old results are
not overwritten. Never delete a live worker's lock.

Validate inputs/dispatch without optimization:

```text
julia --startup-file=no --project=. --threads=1,0 scripts/run_ADD_locality_transfer.jl --check
```

A local two-iteration smoke pair, saved separately:

```text
julia --startup-file=no --project=. --threads=1,0 scripts/run_ADD_locality_transfer.jl --smoke
```

For analysis, pool the nine equally represented parameter combinations within
each w (135 pairs), but also report all 63 cells (15 pairs) and the observed exact
ranks. Do not treat w as rank itself or as an isolated causal intervention: it
also changes the feasible space. Report paired effects and adjusted tests even
when they do not support the proposed rank-dependent advantage.
