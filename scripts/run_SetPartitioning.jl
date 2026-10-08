#!/usr/bin/env julia
# Independent paired comparisons. This reuses the assignment algorithms, rather
# than implementing another variant of EELS or ADD.
include("generate_set_partitioning_instances.jl")
using LinearAlgebra, Statistics
using OhMyU1: SolverParams, SolverStatistics, OptimizationProblem, solve,
    fill_partition_vectors!, sort_and_truncate
module PartitionEELS
    include("run_EELS_optimization.jl")
end

function partition_cost(objectives, objective, m)
    objective == "entropic" || error("Unknown objective: $objective")
    c, W, tau = objectives.c,objectives.scenarios,objectives.tau
    scale = sqrt(m)
    # Stable log-mean-exp of scenario costs. Signed, nonpolynomial, and
    # generally nonseparable even between distant parts of the strip.
    return function(x)
        z = (W*x) ./ (tau*scale)
        peak = maximum(z)
        dot(c,x)/scale + tau*(peak+log(sum(exp(v-peak) for v in z))-log(length(z)))
    end
end

function partition_add_arm(problem,sampler,initial,sp,seed,strategy)
    calls = Ref(0)
    counts = Int[]
    prefix_hashes = String[]
    actual_initial = Ref("")
    function source!(X)
        calls[] += 1
        if calls[] == 1
            copyto!(X,initial)
            trained, _ = sort_and_truncate(copy(X),problem.cost_function,sp.NUM_FEASIBLE_SAMPLES)
            actual_initial[] = PartitionEELS.training_fingerprint(trained)
        else
            # Same stream prefix for both arms at each iteration, independent
            # of model RNG and of how many draws the previous arm requested.
            rng = MersenneTwister(seed+10_000_000*(calls[]-1))
            fill_partition_vectors!(rng,X,sampler)
            push!(counts,size(X,2))
            push!(prefix_hashes,PartitionEELS.training_fingerprint(@view X[:,1:min(32,size(X,2))]))
        end
        X
    end
    Random.seed!(seed)
    start = time_ns()
    stats = solve(problem,source!,sp,strategy;diversify=false,barrier=false,
        mixed_proportion=0.4,rank_weight=0.4,parallel=false,print_stats=false)
    elapsed_seconds = (time_ns()-start)/1e9
    all(problem.A*stats.incub .== problem.b) || error("Infeasible incumbent")
    (; stats,diagnostics=(;elapsed_seconds,initial_training_sha256=actual_initial[],
        external_sample_counts=counts,external_first32_sha256=prefix_hashes))
end

function run_partition_group(root,mode,objective,d,ns,instances,sp,workers)
    path = joinpath(root,"results","res_$(mode)_$(objective)_$(d)_test.jld2")
    mkpath(dirname(path))
    hashes = Dict((n,i)=>bytes2hex(sha256(read(partition_path(root,d,n,i)))) for n in ns for i in instances)
    configuration = (;protocol=2,split="test",mode,objective,density=d,ns=collect(ns),instances=collect(instances),
        input_hashes=hashes,source_hashes=partition_source_hashes(),julia_version=string(VERSION),
        parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(sp,f) for f in fieldnames(SolverParams))),
        gamma=1,add_weight=0.4,diversify_add=false,barrier=false,
        random_column_order=true,objective_tuning=false,
        add_candidate_pool=20000,external_eels_after_initial=0)
    arm = mode == "EELS" ? "EELS" : "best nonzero mixed"
    empty_branch() = Dict(n=>Dict{Int,SolverStatistics}() for n in ns)
    res_dict = Dict{String,Any}("best_cost"=>empty_branch(),
        arm=>(mode == "EELS" ? empty_branch() : Dict(0.4=>empty_branch())))
    diagnostics = Dict{Tuple{Int,Int},Any}()
    if isfile(path)
        saved = load(path)
        saved["configuration"] == configuration || error("Checkpoint/input/code mismatch: $path")
        res_dict,diagnostics = saved["res_dict"],saved["diagnostics"]
    end
    alternative = mode == "EELS" ? res_dict[arm] : res_dict[arm][0.4]
    jobs = [(n,i) for n in ns for i in instances if !(haskey(res_dict["best_cost"][n],i) && haskey(alternative[n],i))]
    isempty(jobs) && return @info "Already complete" path
    queue = Channel{Tuple{Int,Int}}(length(jobs))
    foreach(j->put!(queue,j),jobs)
    close(queue)
    io_lock = ReentrantLock()
    @info "Starting paired comparison" mode objective d pairs=length(jobs) workers
    @sync for _ in 1:min(workers,length(jobs))
        Threads.@spawn for (n,i) in queue
            saved = lock(io_lock) do
                load(partition_path(root,d,n,i))
            end
            p = saved["instance"]
            setup_seconds = @elapsed sampler = ExactPartitionSampler(p.tiles,length(p.b))
            seed = partition_seed(n,i,d)+400_000_000
            initial = zeros(Int,n^2,sp.NUM_FEASIBLE_SAMPLES)
            initial_seconds = @elapsed fill_partition_vectors!(MersenneTwister(seed),initial,sampler)
            all(p.A*initial .== p.b) || error("Infeasible source data")
            initial_hash = PartitionEELS.training_fingerprint(initial)
            problem = OptimizationProblem(A=p.A,b=p.b,
                cost_function=partition_cost(saved["objectives"],objective,length(p.b)),name="partition_$(d)_$(n)_$(i)")
            pair = Dict{String,Any}()
            # Alternate arm order to reduce systematic first-run/JIT timing bias.
            for strategy in (isodd(i) ? ["best_cost",arm] : [arm,"best_cost"])
                if mode == "EELS"
                    stats,diag = PartitionEELS.run_arm(problem,initial,sp,seed;eels=strategy==arm)
                    pair[strategy] = (;stats,diagnostics=diag)
                else
                    pair[strategy] = partition_add_arm(problem,sampler,initial,sp,seed,strategy)
                end
            end
            PartitionEELS.training_fingerprint(initial) == initial_hash || error("Initial data mutated")
            pair["best_cost"].diagnostics.initial_training_sha256 == pair[arm].diagnostics.initial_training_sha256 ||
                error("Paired training data differ")
            if mode == "ADD"
                pair["best_cost"].diagnostics.external_first32_sha256 == pair[arm].diagnostics.external_first32_sha256 ||
                    error("External source streams differ")
            end
            lock(io_lock) do
                res_dict["best_cost"][n][i] = pair["best_cost"].stats
                alternative[n][i] = pair[arm].stats
                diagnostics[(n,i)] = (;seed,initial_sha256=initial_hash,setup_seconds,initial_seconds,
                    density=p.actual_density,exact_cover_count=saved["exact_cover_count"],
                    dp_states=length(sampler.counts),arms=Dict(a=>v.diagnostics for (a,v) in pair))
                jldsave(path*".tmp";res_dict,solver_params=sp,configuration,diagnostics)
                Base.Filesystem.rename(path*".tmp",path)
                @info "Saved pair" mode objective d n i baseline=pair["best_cost"].stats.c_min alternative=pair[arm].stats.c_min
            end
        end
    end
end

function partition_main(args=ARGS)
    allowed = ["--smoke","--eels-only","--add-only","--generate-only"]
    all(a -> a in allowed,args) || error("Options: $(join(allowed, ' '))")
    !("--eels-only" in args && "--add-only" in args) || error("Choose one mode or neither")
    smoke = "--smoke" in args
    root = joinpath(PARTITION_ROOT,"data","set_partitioning",smoke ? "smoke" : "test")
    ns,instances = smoke ? ([12],[1]) : (collect(12:16),collect(1:50))
    densities = ["light"]
    modes = "--eels-only" in args ? ["EELS"] : "--add-only" in args ? ["ADD"] : ["EELS","ADD"]
    BLAS.set_num_threads(1)
    mkpath(root)
    lockdir = joinpath(root,".running")
    ispath(lockdir) && error("Existing run lock: $lockdir. Remove only after verifying that no job is active.")
    mkdir(lockdir)
    try
        write(joinpath(lockdir,"owner.txt"),"pid=$(getpid()) host=$(gethostname())\n")
        generate_partition_instances(root;ns,instances,densities)
        "--generate-only" in args && return
        workers = parse(Int,get(ENV,"PARTITION_WORKERS",string(Threads.nthreads())))
        1 <= workers <= Threads.nthreads() || error("PARTITION_WORKERS must be between 1 and Julia thread count")
        for mode in modes, objective in ["entropic"], d in densities
            sp = SolverParams(NUM_FEASIBLE_SAMPLES=smoke ? 40 : mode == "EELS" ? 400 : 1000,
                NUM_MPS_SAMPLES=smoke ? 80 : 10000,NUM_GLOBAL_ITER=smoke ? 2 : 20,
                NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,LEARNING_RATE=0.05,KEEP_NUM_WORST=0.0)
            run_partition_group(root,mode,objective,d,ns,instances,sp,workers)
        end
    finally
        rm(joinpath(lockdir,"owner.txt");force=true)
        rm(lockdir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    partition_main()
end
