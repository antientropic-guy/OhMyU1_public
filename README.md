# OhMyU1

**OhMyU1** is a Julia package for constrained combinatorial optimization with equality constraints and binary decision vectors. The package implements methods based on matrix product states (MPS) and U(1) symmetry to solve problems of the form:

- minimize `f(x)`
- subject to `A x = b`
- with `x` binary and integer matrix `A`

The approach is inspired by recent research on enhancing combinatorial optimization with classical and quantum generative models.

## Features

- Solver for constrained optimization problems with linear equality constraints
- Support for binary decision variables and arbitrary cost functions
- Matrix Product State (MPS) / U(1) based sampling and training workflows
- Tools for diversification, training, and solver statistics
- Example notebooks for assignment, quadratic assignment and scaling

## Installation

1. Clone the repository:

```bash
git clone https://github.com/antientropic-guy/OhMyU1_public.git
cd OhMyU1_public
```

2. Start Julia in the project environment and install dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Use the package from the local workspace:

```julia
using Pkg
Pkg.develop(path=".")
using OhMyU1
```

## Folder structure

- `src/`
  - `OhMyU1.jl` - main module entry point and package definition
  - `solver.jl` - solver parameters, optimization problem types, training loops, and statistics
  - `mps_core.jl` - core MPS and U(1) tensor network implementation
  - `tools.jl` - helper functions and utilities
  - `diversification.jl` - diversification and sample selection methods
  - `train.jl` - training routines for MPS-based optimization
  - `future.jl` - planned extensions for non-binary variables and higher TT ranks
  - `deprecated.jl` - deprecated code
  - `py_tools.jl` - Python interoperability utilities (deprecated)

- `data/`
  - `assignment_results/` - stored results and parameters for assignment experiments
  - `random_assignment/` - random assignment problem parameters and results
  - `random_bilinear_forms/` - random bilinear forms for testing
  - `random_problems_*` - collections of synthetic random problem instances (n - number of variables, m - number of constraints, r - parameter of coefficients uniform distribution in range [-r, r])
  - `random_vectors/` - random vector datasets used by experiments (used as linear cost functions, r is a parameter of coefficients uniform distribution in range [-r, r])

- `examples/`
  - `LinearAssignmentReport.ipynb` - notebook demonstrating linear assignment results and Wilcoxon signed-rank test for them (ADD)
  - `QuadraticAssignmentReport.ipynb` - quadratic assignment results and Wilcoxon signed-rank test for them (ADD)
  - `ResultsNumSamples.ipynb` - analysis of unique samples counts and runtime (EELS)
  - `Scaling.ipynb` - scaling experiments and performance studies
  - `pictures/` - image assets

- `scripts/`

This folder contains scripts used for experiments in the article. You may easily open any script and modify parameters their, reproducing paper results on your machine, or conducting your own experiments. Results for the experiments are stored as nested dictionaries, for example, look in the data folder.

  - `run_BestCost.jl` - script to run best-cost baseline
  - `run_EELS_assignment.jl` - script for benchmarking EELS algorithm on assignment problems
  - `run_EELS_random_problems.jl` - script for benchmarking EELS algorithm on random problem
  - `run_Strategy.jl` - script for ADD strategy evaluation

- `Project.toml` / `Manifest.toml` - Julia project dependencies and exact package versions

## Notes

- The current implementation focuses on binary variables and supports degeneracy rank 1 workflows.
- The `future.jl` file contains planned generic methods for non-binary variables and higher degeneracies.
- The repository includes experimental notebooks and prepared datasets for benchmarking and analysis.
Due to the refactoring of the folder structure, you may encounter some issues with paths to raw data or cached results. In such cases, check the paths (all relevant data is located in the `data` folder), or contact sergeyyoudin@gmail.com

## EELS objective-value experiment

Submit the fixed `gamma=1` comparison on the cluster with:

```bash
bash scripts/bash/run_EELS_optimization.sh
```

The launcher submits a Slurm job using the existing `c32m384` partition,
32 CPUs and a 24-hour limit. It locates the project automatically and uses
its pinned Julia environment (Julia 1.12). Julia must be available in `PATH`,
as in the other cluster scripts. No edits or weight grid are needed.

`scripts/run_EELS_optimization.jl` compares `best_cost` (Method (ii)) and
`EELS` on the existing linear and quadratic assignment **test** instances
only: `n=12:16`, instance IDs `1:50`. No train/validation instances are used.
Each pair starts from the
same 400 feasible draws, then performs 20 iterations of one training sweep
(learning rate 0.05) and 10,000 TT draws. Both arms retain up to 400 best
unique candidates from the previous training set and the new TT samples.
There are **no additional external feasible draws after initialization and
no ADD selection**. Consequently the baseline is rerun in this scarce-data
protocol; existing ADD baseline results are not reused as matched controls.

Results are checkpointed after each completed pair in
`data/eels_optimization/gamma1/res_EELS_gamma1_{linear,quadratic}_test.jld2`.
Each file contains `res_dict`, `solver_params`, `configuration`, and
`diagnostics`. The result dictionary has the same non-parametric layout as
`run_BestCost.jl`, with no artificial weight dimension:

```julia
using OhMyU1, JLD2
d = load("data/eels_optimization/gamma1/res_EELS_gamma1_quadratic_test.jld2")
baseline = d["res_dict"]["best_cost"][12][1] # SolverStatistics
eels = d["res_dict"]["EELS"][12][1]
eels.c_min              # final best objective value
eels.learning_curve     # initial value followed by 20 iterations
eels.incub              # best assignment vector
d["diagnostics"]["EELS"][(12, 1)].evaluations # cumulative unique oracle calls
```

Seeds, input-file hashes and parameters are recorded. Each newly computed pair
checks that the shared initial matrix was not mutated and that both arms used
the same initial training matrix after deduplication and sorting; its SHA-256
is stored in each arm's diagnostics as `initial_training_sha256`.
The sampling budget is
matched; unique objective evaluations may differ and are counted separately.
`utility_curve` measures the mean lowest 5% (at least one) of the unique
selection pool. Elapsed times include compilation when first encountered;
they are diagnostics, not a warmed-up kernel benchmark.
Relaunching skips completed pairs and rejects incompatible checkpoints.
After a forcibly killed job, verify that it has stopped and remove the stale
`data/eels_optimization/gamma1/.running` directory before resubmitting.

For a short check on existing `n=12, i=1` instances, use
`bash scripts/bash/run_EELS_optimization.sh --smoke`; its results go into
the separate `data/eels_optimization/smoke/` directory. Local regression
checks run with `julia --project=. --threads=2 scripts/test_EELS_optimization.jl`.

## Compatibility

The package dependencies are specified in `Project.toml`. It is tested with Julia-compatible versions of packages such as `JuMP`, `GLPK`, `SCIP`, `Plots`, and `JLD2`.

## Citation
### BibTex record:
@inproceedings{
iudin2026explorationexploitation,
title={Exploration-Exploitation Generative Framework for Constrained Combinatorial Optimization with Tensor Trains},
author={Sergei Iudin and Mikhail Podobrii and Dmitry Zheltkov and Stanislav Moiseev},
booktitle={ICML'26 workshop on CoLoRAI - The 2nd Workshop on Connecting Low-rank Representations in AI},
year={2026},
url={https://openreview.net/forum?id=57YDrXbZuD}
}

## Contacts
With comments, questions, and suggestions, please contact sergeyyoudin@gmail.com
