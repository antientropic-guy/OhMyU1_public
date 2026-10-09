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

const PARTITION_ADD_STRATEGY = "best nonzero mixed" # Article: Best Cost + GDF
const PARTITION_ADD_WEIGHT = 0.4                    # Article: w_filter
const PARTITION_PROTOCOL = 3                       # Isolated per-pair workers
partition_results_dir(root) = joinpath(root,"results_v3")

function partition_solver_params(mode; smoke=false)
    mode in ("ADD","EELS") || error("Unknown mode: $mode")
    SolverParams(NUM_FEASIBLE_SAMPLES=smoke ? 40 : mode == "EELS" ? 400 : 1000,
        NUM_MPS_SAMPLES=smoke ? 80 : 10000,NUM_GLOBAL_ITER=smoke ? 2 : 20,
        NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,LEARNING_RATE=0.05,
        UTILITY_FRACTION=0.05,KEEP_NUM_WORST=0.0)
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
        mixed_proportion=PARTITION_ADD_WEIGHT,rank_weight=PARTITION_ADD_WEIGHT,parallel=false,print_stats=false)
    elapsed_seconds = (time_ns()-start)/1e9
    all(problem.A*stats.incub .== problem.b) || error("Infeasible incumbent")
    (; stats,diagnostics=(;elapsed_seconds,initial_training_sha256=actual_initial[],
        external_sample_counts=counts,external_first32_sha256=prefix_hashes))
end

function run_partition_pair(input_data,mode,objective,d,n,i,sp,input_sha256)
    # Function arguments/local bindings cannot be captured from the dispatcher.
    # Each invocation owns its input, sampler, initial data, objective and stats.
    spec = input_data["specification"]
    (spec.n,spec.i,spec.density,spec.split) == (n,i,d,"test") || error("Wrong input assigned to worker")
    p = input_data["instance"]
    size(p.A) == (3n,n^2) || error("Wrong instance dimensions")
    setup_seconds = @elapsed sampler = ExactPartitionSampler(p.tiles,length(p.b))
    sampler.counts[UInt128(0)] == input_data["exact_cover_count"] || error("Input cover count mismatch")
    seed = partition_seed(n,i,d)+400_000_000
    initial = zeros(Int,n^2,sp.NUM_FEASIBLE_SAMPLES)
    initial_seconds = @elapsed fill_partition_vectors!(MersenneTwister(seed),initial,sampler)
    all(p.A*initial .== p.b) || error("Infeasible source data")
    initial_hash = PartitionEELS.training_fingerprint(initial)
    problem = OptimizationProblem(A=p.A,b=p.b,
        cost_function=partition_cost(input_data["objectives"],objective,length(p.b)),name="partition_$(d)_$(n)_$(i)")
    arm = mode == "EELS" ? "EELS" : PARTITION_ADD_STRATEGY
    pair = Dict{String,Any}()
    for strategy in (isodd(i) ? ["best_cost",arm] : [arm,"best_cost"])
        if mode == "EELS"
            stats,diag = PartitionEELS.run_arm(problem,initial,sp,seed;eels=strategy==arm)
            pair[strategy] = (;stats,diagnostics=diag)
        else
            pair[strategy] = partition_add_arm(problem,sampler,initial,sp,seed,strategy)
        end
    end
    PartitionEELS.training_fingerprint(initial) == initial_hash || error("Initial data mutated")
    for result in values(pair)
        all(p.A*result.stats.incub .== p.b) || error("Infeasible saved incumbent")
        isapprox(problem.cost_function(result.stats.incub),result.stats.c_min;atol=1e-10,rtol=1e-10) ||
            error("Saved cost does not match this instance's objective")
    end
    pair["best_cost"].diagnostics.initial_training_sha256 == pair[arm].diagnostics.initial_training_sha256 ||
        error("Paired training data differ")
    if mode == "ADD"
        pair["best_cost"].diagnostics.external_first32_sha256 == pair[arm].diagnostics.external_first32_sha256 ||
            error("External source streams differ")
    end
    diagnostics = (;seed,initial_sha256=initial_hash,input_sha256,setup_seconds,initial_seconds,
        density=p.actual_density,exact_cover_count=input_data["exact_cover_count"],
        dp_states=length(sampler.counts),arms=Dict(a=>v.diagnostics for (a,v) in pair))
    (;baseline=pair["best_cost"].stats,alternative=pair[arm].stats,diagnostics)
end

function run_partition_group(root,mode,objective,d,ns,instances,sp,workers;
        result_directory=partition_results_dir(root))
    path = joinpath(result_directory,"res_$(mode)_$(objective)_$(d)_test.jld2")
    mkpath(dirname(path))
    hashes = Dict((n,i)=>bytes2hex(sha256(read(partition_path(root,d,n,i)))) for n in ns for i in instances)
    configuration = (;protocol=PARTITION_PROTOCOL,split="test",mode,objective,density=d,ns=collect(ns),instances=collect(instances),
        input_hashes=hashes,source_hashes=partition_source_hashes(),julia_version=string(VERSION),
        parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(sp,f) for f in fieldnames(SolverParams))),
        gamma=1,add_strategy=PARTITION_ADD_STRATEGY,add_weight=PARTITION_ADD_WEIGHT,diversify_add=false,barrier=false,
        random_column_order=true,objective_tuning=false,
        add_candidate_pool=20000,external_eels_after_initial=0)
    arm = mode == "EELS" ? "EELS" : PARTITION_ADD_STRATEGY
    empty_branch() = Dict(n=>Dict{Int,SolverStatistics}() for n in ns)
    res_dict = Dict{String,Any}("best_cost"=>empty_branch(),
        arm=>(mode == "EELS" ? empty_branch() : Dict(PARTITION_ADD_WEIGHT=>empty_branch())))
    diagnostics = Dict{Tuple{Int,Int},Any}()
    if isfile(path)
        checkpoint = load(path)
        checkpoint["configuration"] == configuration || error("Checkpoint/input/code mismatch: $path")
        res_dict,diagnostics = checkpoint["res_dict"],checkpoint["diagnostics"]
    end
    alternative = mode == "EELS" ? res_dict[arm] : res_dict[arm][PARTITION_ADD_WEIGHT]
    jobs = [(n,i) for n in ns for i in instances if !(haskey(res_dict["best_cost"][n],i) && haskey(alternative[n],i))]
    isempty(jobs) && return @info "Already complete" path
    queue = Channel{Tuple{Int,Int}}(length(jobs))
    foreach(j->put!(queue,j),jobs)
    close(queue)
    io_lock = ReentrantLock()
    @info "Starting paired comparison" mode objective d pairs=length(jobs) workers
    @sync for _ in 1:min(workers,length(jobs))
        Threads.@spawn for (n,i) in queue
            # Explicitly task-local: never reuse a binding assigned in the
            # enclosing function (checkpoint loading used to share `saved`).
            local input_data = lock(io_lock) do
                input_path = partition_path(root,d,n,i)
                bytes2hex(sha256(read(input_path))) == hashes[(n,i)] || error("Input file changed")
                load(input_path)
            end
            local completed = run_partition_pair(input_data,mode,objective,d,n,i,sp,hashes[(n,i)])
            lock(io_lock) do
                # Reload the declared instance independently of worker bindings.
                declared = load(partition_path(root,d,n,i))
                declared_cost = partition_cost(declared["objectives"],objective,length(declared["instance"].b))
                for stats in (completed.baseline,completed.alternative)
                    all(declared["instance"].A*stats.incub .== declared["instance"].b) ||
                        error("Independent pre-save feasibility audit failed: n=$n i=$i")
                    isapprox(declared_cost(stats.incub),stats.c_min;atol=1e-10,rtol=1e-10) ||
                        error("Independent pre-save objective audit failed: n=$n i=$i")
                end
                completed.diagnostics.exact_cover_count == declared["exact_cover_count"] || error("Wrong metadata")
                res_dict["best_cost"][n][i] = completed.baseline
                alternative[n][i] = completed.alternative
                diagnostics[(n,i)] = completed.diagnostics
                jldsave(path*".tmp";res_dict,solver_params=sp,configuration,diagnostics)
                Base.Filesystem.rename(path*".tmp",path)
                @info "Saved pair" mode objective d n i baseline=completed.baseline.c_min alternative=completed.alternative.c_min
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
            sp = partition_solver_params(mode;smoke)
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
