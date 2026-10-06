#!/usr/bin/env julia

# Paired, scarce-data EELS experiment on existing TEST assignment objectives.
# Run via scripts/bash/run_EELS_optimization.sh (Slurm), or:
# julia --project=. --threads=auto scripts/run_EELS_optimization.jl
# --smoke uses existing n=12, i=1 data and a separate output directory.
# No tuning: gamma=1, Best Cost selection in BOTH arms, no external samples
# after initialization. This deliberately does not call solve(), which injects
# fresh feasible samples every iteration in the ADD experiments.

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)
using OhMyU1, JLD2, Random, LinearAlgebra, Statistics, SHA
using OhMyU1: SolverParams, SolverStatistics, OptimizationProblem,
    assignment_matrix, fill_assignment_vectors!, build_mps_from_feasible_samples,
    orthogonalize!, normalize!, u1_norm, TrainParams, train_nondeg!, sample_nondeg!

const ResultDict = Dict{String,Dict{Int,Dict{Int,SolverStatistics}}}
const ARMS = ("best_cost", "EELS")

function instance_path(split, objective, n, i)
    split == "test" || error("This experiment uses only held-out test instances")
    prefix = "test_"
    suffix = "_test"
    folder, stem, radius = objective == "linear" ?
        ("random_vectors", "c", 50) : ("random_bilinear_forms", "q", 7)
    joinpath(ROOT, "data", prefix * folder, "$(stem)_$(n)_$(i)_r=$(radius)$(suffix).jld2")
end

# Explicit integer seeds, independent of thread scheduling and Julia hash salt.
pair_seed(split, objective, n, i) = 20261006 +
    1_000_000 * (split == "test") + 100_000 * (objective == "quadratic") + 100n + i

# Binary training matrix fingerprint includes dimensions and column order.
function training_fingerprint(T)
    io = IOBuffer()
    print(io, size(T), ':')
    write(io, UInt8.(vec(T)))
    bytes2hex(sha256(take!(io)))
end

function run_arm(problem, initial, sp, seed; eels=false, on_initial_training=nothing)
    cache = Dict{Tuple,Float64}()
    cached_cost(x) = get!(cache, Tuple(x)) do
        value = Float64(problem.cost_function(x))
        isfinite(value) || error("Non-finite objective value")
        value
    end
    T = unique(copy(initial); dims=2)
    stats = SolverStatistics()
    evaluations = Int[]
    support_sizes = Float64[]
    training_sizes = Int[]
    elapsed_seconds = Float64[]
    start = time_ns()

    function record!(data, costs)
        order = sortperm(costs)
        elite_count = max(1, floor(Int, sp.UTILITY_FRACTION * length(costs)))
        stats.c_min = costs[first(order)]
        stats.incub = copy(data[:, first(order)])
        push!(stats.learning_curve, stats.c_min)
        push!(stats.utility_curve, mean(costs[order[1:elite_count]]))
        push!(evaluations, length(cache))
        push!(elapsed_seconds, (time_ns() - start) / 1e9)
        order
    end

    costs = [cached_cost(x) for x in eachcol(T)]
    order = record!(T, costs)
    T = T[:, order]
    costs = costs[order]
    initial_training_sha256 = training_fingerprint(T)
    # Optional observer used by regression checks to inspect the exact matrix
    # after deduplication/sorting and before model construction/training.
    isnothing(on_initial_training) || on_initial_training(copy(T))
    for iteration in 1:sp.NUM_GLOBAL_ITER
        push!(training_sizes, size(T, 2))
        # diversify! is the existing fixed gamma=1 implementation. Apply it
        # BEFORE canonicalization; all allowed initial amplitudes are one.
        mps = build_mps_from_feasible_samples(problem, T, 1, eels)
        orthogonalize!(mps)
        support = Float64(u1_norm(mps)^2)
        push!(support_sizes, support)
        stats.num_samples_in_mps = support < typemax(Int) ? round(Int, support) : -1
        normalize!(mps)
        temperature = length(costs) > 1 ? std(costs) : 0.0
        weights = temperature > 0 ? exp.(-(costs .- minimum(costs)) ./ temperature) : ones(length(costs))
        weights ./= sum(weights)
        train_nondeg!(mps, sp.NUM_SWEEP_ITER, T, weights, TrainParams(sp.LEARNING_RATE, 1))
        isfinite(u1_norm(mps)) && u1_norm(mps) > 0 || error("Invalid trained norm")
        normalize!(mps)
        stats.temperature = temperature

        # Reset each iteration so both arms use the same sampling seed, even
        # after their distributions diverge. Sampling stays serial inside a pair.
        Random.seed!(seed + 10_000_000iteration)
        samples = zeros(Int, problem.num_vars, sp.NUM_MPS_SAMPLES)
        sample_nondeg!(mps, samples, sp.NUM_MPS_SAMPLES)
        all(problem.A * samples .== problem.b) || error("Infeasible generated sample")
        stats.num_unique_samples = size(unique(samples; dims=2), 2)
        # Keep the previous training set in the selection pool: the incumbent
        # cannot be lost, and a collapsed sampler cannot erase all training data.
        pool = unique(hcat(T, samples); dims=2)
        pool_costs = [cached_cost(x) for x in eachcol(pool)]
        order = record!(pool, pool_costs)
        selected = order[1:min(sp.NUM_FEASIBLE_SAMPLES, length(order))]
        T, costs = pool[:, selected], pool_costs[selected]
    end
    stats.num_to_keep_worst = 0
    diagnostics = (; evaluations, support_sizes, training_sizes, elapsed_seconds, initial_training_sha256,
        utility_definition="mean lowest 5% (at least one) of unique candidates plus retained training data")
    return stats, diagnostics
end

function run_group(split, objective, ns, instances, sp, output)
    mkpath(output)
    path = joinpath(output, "res_EELS_gamma1_$(objective)_$(split).jld2")
    jobs = [(n, i) for n in ns for i in instances]
    # Check ALL inputs before launching workers; never regenerate held-out data.
    hashes = Dict{Tuple{Int,Int},String}()
    for (n, i) in jobs
        source = instance_path(split, objective, n, i)
        isfile(source) || error("Missing existing instance: $source")
        hashes[(n, i)] = bytes2hex(sha256(read(source)))
    end
    configuration = (; protocol=1, split, objective, gamma=1, ns=collect(ns),
        instances=collect(instances), hashes,
        parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(sp, f) for f in fieldnames(SolverParams))),
        seed_rule="20261006 + 1000000*(test) + 100000*(quadratic) + 100*n + i",
        external_samples_after_initialization=0, julia_version=string(VERSION))
    res_dict = ResultDict(arm => Dict(n => Dict{Int,SolverStatistics}() for n in ns) for arm in ARMS)
    diagnostics = Dict{String,Dict{Tuple{Int,Int},Any}}(arm => Dict() for arm in ARMS)
    if isfile(path)
        saved = load(path)
        saved["configuration"] == configuration || error("Checkpoint configuration/input mismatch: $path")
        res_dict, diagnostics = saved["res_dict"], saved["diagnostics"]
    end
    pending = [(n, i) for (n, i) in jobs if !all(haskey(res_dict[a][n], i) for a in ARMS)]
    isempty(pending) && return println("Already complete: $path")
    queue = Channel{Tuple{Int,Int}}(length(pending))
    foreach(job -> put!(queue, job), pending)
    close(queue)
    io_lock = ReentrantLock()
    workers = min(Threads.nthreads(), length(pending))
    @info "Starting EELS comparison" split objective workers pending=length(pending)
    @sync for _ in 1:workers
        Threads.@spawn for (n, i) in queue
            seed = pair_seed(split, objective, n, i)
            value = lock(io_lock) do
                load(instance_path(split, objective, n, i), objective == "linear" ? "c" : "q")
            end
            expected_size = objective == "linear" ? (n^2,) : (n^2, n^2)
            size(value) == expected_size || error("Incorrect objective shape for n=$n, i=$i")
            cost = objective == "linear" ? x -> dot(value, x) : x -> dot(x, value * x)
            problem = OptimizationProblem(A=assignment_matrix(n), b=ones(Int, 2n), cost_function=cost)
            Random.seed!(seed)
            initial = zeros(Int, n^2, sp.NUM_FEASIBLE_SAMPLES)
            fill_assignment_vectors!(initial, n)
            initial_sha256 = training_fingerprint(initial)
            pair = Dict(arm => run_arm(problem, initial, sp, seed; eels=(arm == "EELS")) for arm in ARMS)
            training_fingerprint(initial) == initial_sha256 || error("Shared initial data were mutated")
            pair["best_cost"][2].initial_training_sha256 == pair["EELS"][2].initial_training_sha256 ||
                error("Initial training data differ between paired strategies")
            lock(io_lock) do
                for arm in ARMS
                    res_dict[arm][n][i], diagnostics[arm][(n, i)] = pair[arm]
                end
                # Atomic replacement on the same filesystem. Only completed
                # pairs are checkpointed; interrupted pairs are rerun on resume.
                temporary = path * ".tmp"
                jldsave(temporary; res_dict, solver_params=sp, configuration, diagnostics)
                Base.Filesystem.rename(temporary, path)
                @info "Saved pair" split objective n i baseline=pair["best_cost"][1].c_min eels=pair["EELS"][1].c_min
            end
        end
    end
    path
end

function main(args=ARGS)
    all(a -> a == "--smoke", args) || error("Usage: run_EELS_optimization.jl [--smoke]")
    smoke = "--smoke" in args
    BLAS.set_num_threads(1)
    sp = SolverParams(NUM_FEASIBLE_SAMPLES=smoke ? 20 : 400,
        NUM_MPS_SAMPLES=smoke ? 40 : 10000, NUM_GLOBAL_ITER=smoke ? 2 : 20,
        NUM_SWEEP_ITER=1, LINK_DEGENERACY=1, LEARNING_RATE=0.05, KEEP_NUM_WORST=0.0)
    output = joinpath(ROOT, "data", "eels_optimization", smoke ? "smoke" : "gamma1")
    mkpath(output)
    # Prevent simultaneous writers (also when launching Julia without Bash).
    lockdir = joinpath(output, ".running")
    ispath(lockdir) && error("Run lock exists: $lockdir. If an earlier job was killed, verify it has stopped before removing this directory.")
    mkdir(lockdir)
    try
        for objective in ("linear", "quadratic")
            run_group("test", objective, smoke ? (12:12) : (12:16), smoke ? (1:1) : (1:50), sp, output)
        end
    finally
        rm(lockdir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
