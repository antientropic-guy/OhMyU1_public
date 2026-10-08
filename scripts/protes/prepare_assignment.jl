#!/usr/bin/env julia
# Exact Julia-side export: never reinterpret Julia RNG seeds in Python.
include(joinpath(@__DIR__, "..", "scripts", "run_EELS_optimization.jl"))
using DelimitedFiles

function option(name, default)
    p = findfirst(==(name), ARGS)
    isnothing(p) ? default : ARGS[p+1]
end
parse_range(s) = occursin(':', s) ? (a = parse.(Int, split(s, ':')); a[1]:a[2]) : [parse(Int, s)]

function prepare()
    Threads.nthreads() == 1 || error("Run Julia with --threads=1")
    BLAS.set_num_threads(1)
    output = abspath(option("--output", joinpath(@__DIR__, "..", "data", "protes_comparison")))
    ns = parse_range(option("--sizes", "12:16"))
    ids = parse_range(option("--instances", "1:50"))
    # 'all' gives an exact per-instance wall-time comparator on the server.
    timing = option("--timing", "all")
    timing in ("all", "first", "none") || error("Invalid timing mode")
    sp = SolverParams(NUM_FEASIBLE_SAMPLES=400, NUM_MPS_SAMPLES=10000,
        NUM_GLOBAL_ITER=20, NUM_SWEEP_ITER=1, LINK_DEGENERACY=1,
        LEARNING_RATE=0.05, KEEP_NUM_WORST=0.0)
    warmsp = SolverParams(NUM_FEASIBLE_SAMPLES=400, NUM_MPS_SAMPLES=100,
        NUM_GLOBAL_ITER=1, NUM_SWEEP_ITER=1, LINK_DEGENERACY=1,
        LEARNING_RATE=0.05, KEEP_NUM_WORST=0.0)
    mkpath(output)
    objectives = split(option("--objectives", "linear,quadratic"), ',')
    all(in(("linear", "quadratic")), objectives) || error("Invalid objective")
    for objective in objectives, n in ns, i in ids
        folder = joinpath(output, "inputs", "$(objective)_$(n)_$(i)")
        mkpath(folder)
        source = instance_path("test", objective, n, i)
        value = load(source, objective == "linear" ? "c" : "q")
        seed = pair_seed("test", objective, n, i)
        Random.seed!(seed)
        initial = zeros(Int, n^2, 400)
        fill_assignment_vectors!(initial, n)
        cost = objective == "linear" ? x -> dot(value, x) : x -> dot(x, value*x)
        problem = OptimizationProblem(A=assignment_matrix(n), b=ones(Int, 2n), cost_function=cost)
        all(problem.A * initial .== problem.b) || error("Invalid initial samples")
        open(joinpath(folder, "objective.f64"), "w") do io
            write(io, Float64.(vec(value)))
        end
        open(joinpath(folder, "initial.u8"), "w") do io
            write(io, UInt8.(vec(initial)))
        end
        writedlm(joinpath(folder, "initial_costs.csv"), [cost(x) for x in eachcol(initial)], ',')
        open(joinpath(folder, "metadata.tsv"), "w") do io
            println(io, "objective\tn\tinstance\tseed\tinitial_sha256\tsource_sha256")
            println(io, join((objective, n, i, seed, training_fingerprint(initial), bytes2hex(sha256(read(source)))), '\t'))
        end
        measured = timing == "all" || (timing == "first" && i == first(ids))
        measured || continue
        # Resume only results accompanied by the exact exported dataset and source.
        savedpath = joinpath(folder, "eels.jld2")
        if isfile(savedpath)
            saved = load(savedpath)
            expected = training_fingerprint(unique(initial; dims=2)[:, sortperm([cost(x) for x in eachcol(unique(initial; dims=2))])])
            saved["diagnostics"].initial_training_sha256 == expected || error("Checkpoint initial data mismatch")
            saved["source_sha256"] == bytes2hex(sha256(read(source))) || error("Checkpoint objective mismatch")
            @info "Reusing EELS timing" objective n i
            continue
        end
        @info "Warming EELS (excluded from measured runtime)" objective n i
        warmstart = time_ns()
        run_arm(problem, initial, warmsp, seed; eels=true)
        warmseconds = (time_ns()-warmstart)/1e9
        GC.gc()
        @info "Timing full EELS on one CPU: 400 initial, 20 x 10000 samples" objective n i
        start = time_ns()
        stats, diagnostics = run_arm(problem, initial, sp, seed; eels=true)
        seconds = (time_ns()-start)/1e9
        jldsave(joinpath(folder, "eels.jld2"); stats, diagnostics, sp, seconds, warmseconds,
            source_sha256=bytes2hex(sha256(read(source))))
        open(joinpath(folder, "eels_timing.tsv"), "w") do io
            println(io, "seconds\twarmup_seconds\tc_min\tevaluations\tinitial_training_sha256")
            println(io, join((seconds, warmseconds, stats.c_min, last(diagnostics.evaluations), diagnostics.initial_training_sha256), '\t'))
        end
        writedlm(joinpath(folder, "eels_curve.csv"), hcat(diagnostics.elapsed_seconds, diagnostics.evaluations, stats.learning_curve), ',')
        @info "EELS measured" objective n i seconds c_min=stats.c_min
        flush(stdout)
    end
end
prepare()
