# ConstrainTNet benchmark environment

The 2026-10-12 assignment benchmark uses upstream ConstrainTNet commit
`77d8c26357520c1869252745ec302da22f21793d` from
https://github.com/JaviLoPiq/ConstrainTNet.jl.
Its algorithm sources are unmodified. The only source change in its vendored
ITensors is `maxQNs = 20` to `maxQNs = 21`. At 20 constraints the upstream
comparison of a full charge with `QN()` attempts to insert an inactive charge
component and fails. One spare slot avoids that boundary error. The previous
local setting of 500 and the local gradient accumulator patch were not used.

These two TOML files record the dependency environment used with Julia 1.12.5.
Compared with the upstream project, the unused SCIP dependency is removed,
Plots compatibility is set to 1.41.6, and the manifest is resolved for the
current Julia version. Vendored ITensors 0.3.41 and NDTensors 0.1.50 are
retained. Local paths in the manifest are relative to the clone root.

To reproduce from the OhMyU1 repository root in PowerShell:

```powershell
git clone https://github.com/JaviLoPiq/ConstrainTNet.jl.git tmp/constraintnet_benchmark
git -C tmp/constraintnet_benchmark checkout --detach 77d8c26357520c1869252745ec302da22f21793d
Copy-Item examples/constraintnet_benchmark_environment/Project.toml tmp/constraintnet_benchmark/Project.toml
Copy-Item examples/constraintnet_benchmark_environment/Manifest.toml tmp/constraintnet_benchmark/Manifest.toml
$qnPath = (Resolve-Path tmp/constraintnet_benchmark/ITensors.jl/src/qn/qn.jl).Path
$qnText = [IO.File]::ReadAllText($qnPath)
if (-not $qnText.Contains('const maxQNs = 20')) { throw 'Unexpected upstream maxQNs' }
[IO.File]::WriteAllText($qnPath, $qnText.Replace('const maxQNs = 20', 'const maxQNs = 21'))
julia --startup-file=no --project=tmp/constraintnet_benchmark -e 'using Pkg; Pkg.instantiate()'
$env:CONSTRAINTNET_BENCHMARK_PROJECT = (Resolve-Path tmp/constraintnet_benchmark).Path
julia --startup-file=no --threads=1,0 --project=. scripts/benchmark_assignment.jl
julia --startup-file=no --project=. scripts/plot_assignment_benchmark.jl
```

The driver launches the backends sequentially, each with one default Julia
thread, no interactive thread, and one BLAS thread. Stop competing local
computations before running. The reported metric excludes compilation,
initial-state construction and preparation, and copies. It includes each
public training routine's internal work. OhMyU1 uses two-site updates;
ConstrainTNet uses one-site updates, environment construction, and loss
evaluation. Equal sweep counts do not imply equal optimization progress.

The plot preserves the previous colors, markers, and axes, except that the
lower time limit is extended to 0.001 seconds to include the faster results.
