# Assignment: persistent BestCost versus EELS

Launch from the repository root on the cluster:

```bash
bash scripts/bash/run_EELS_optimization.sh
```

The launcher submits one Slurm job. Its resource settings are unchanged.
For a small separate smoke run, append `--smoke`.

Protocol 3 in `run_EELS_optimization.jl` follows Algorithm 1:

1. Generate one shared initial batch of 400 feasible assignment solutions.
   Deduplicate it and construct each model once. Apply gamma=1 EELS once in
   the EELS arm; use Method (ii) in the BestCost arm. Normalize the initially
   uniform models.
2. Sample 10,000 points from the current model using the legacy sampler.
3. Deduplicate the batch, evaluate costs, and select its best at most 400
   distinct points as the new training set. If fewer than 400 are available,
   use all of them. Do not add initial data or historical elites to this set.
4. Fit the SAME model using normalized Boltzmann weights, temperature equal
   to the sample standard deviation of current elite costs, one full
   sweep, and learning rate 0.05. Equal costs give uniform target weights.
5. Repeat steps 2--4 for 20 iterations. There is no preliminary training step,
   amplitude reset, repeated EELS expansion, or external data injection.

The charge topology remains fixed after initialization. Each training set
contains at most 400 points; learned amplitudes carry information forward.
The best-ever solution is stored independently and can survive its absence
from the current training set. UTILITY_FRACTION=0.05 only defines the reported
utility statistic; NUM_FEASIBLE_SAMPLES=400 determines the elite capacity.
The final trained model is not sampled again, consistent with the loop order
in Algorithm 1. The returned incumbent is the best evaluated feasible point.

Both linear and quadratic held-out objectives use n=12 through 16, instances
1 through 50. Initial batches and iteration sampling seeds are paired between
the arms. Tasks own their model, cache, and RNG state. Checkpoint writes are
serialized and completed pairs are resumable.

Results retain the existing `res_dict[arm][n][instance]` format, saved under
`data/eels_optimization/gamma1_persistent_v3/`. Smoke results go under
`smoke_persistent_v3/`. Old results are preserved. Configuration metadata
records protocol 3 and rejects incompatible checkpoints. Existing analysis
notebooks still refer to the old results until explicitly updated.

`run_persistent_arm` implements this protocol. The old `run_arm` helper is
retained solely to preserve existing Set Partitioning and PROTES callers;
the assignment cluster entry point does not call it. SA experiments are
unaffected.

Run regression checks in Windows PowerShell:

```powershell
julia --startup-file=no --project=. --threads=4,1 scripts/test_EELS_optimization.jl
```

Checks cover paired RNG data, feasible incumbents, resumable concurrent
checkpoints, equal-cost targets, initial-input immutability, exact bounded
elite selection from the current batch, loop order, model identity, and preservation of trained
core entries until sampling in the next iteration.
