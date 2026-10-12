#!/usr/bin/env julia
# Four independent strategy jobs, training objectives only, bounded live solves.
using Pkg
const ROOT=normpath(joinpath(@__DIR__,".."))
Pkg.activate(ROOT)
using OhMyU1, Random, LinearAlgebra, JLD2, SHA
const U=OhMyU1
const STRATEGIES=Dict("best_cost"=>"best_cost","mixed"=>"mixed",
    "weighted_rank"=>"weighted rank","gdf"=>"best nonzero mixed")
# Original article screening grids, recorded in LinearAssignmentReport.ipynb
# and QuadraticAssignmentReport.ipynb. The manuscript tables show winners only.
const WEIGHT_GRIDS=Dict("best_cost"=>[0.],"mixed"=>[.1,.2,.3,.4],
    "weighted_rank"=>[.1,.35,.5,.7],"gdf"=>[.1,.2,.3,.4])
function options(args)
    opt=Dict("strategy"=>"","objective"=>"both","workers"=>"4","output"=>"")
    smoke=false;check=false;i=1
    while i<=length(args)
        a=args[i]
        if a=="--smoke";smoke=true
        elseif a=="--check";check=true
        elseif startswith(a,"--") && haskey(opt,a[3:end])
            i+=1;i<=length(args) || error("Missing value for $a");opt[a[3:end]]=args[i]
        else;error("Unknown option: $a")
        end
        i+=1
    end
    haskey(STRATEGIES,opt["strategy"]) || error("Required --strategy: best_cost, mixed, weighted_rank, gdf")
    opt["objective"] in ("both","linear","quadratic") || error("Invalid objective")
    parse(Int,opt["workers"])>0 || error("Workers must be positive")
    return opt,smoke,check
end
input_path(objective,n,i)= objective=="linear" ?
    joinpath(ROOT,"data","random_vectors","c_$(n)_$(i)_r=50.jld2") :
    joinpath(ROOT,"data","random_bilinear_forms","q_$(n)_$(i)_r=7.jld2")
function implementation_hash()
    files=vcat([joinpath(ROOT,"src",f) for f in sort(readdir(joinpath(ROOT,"src"))) if endswith(f,".jl")],[@__FILE__])
    bytes2hex(sha256(join(bytes2hex(sha256(read(f))) for f in files)))
end
# Different strategy/weight jobs receive the same initial examples. The source
# owns an RNG, so model sampling cannot advance it. Subsequent requested batch
# sizes can differ between strategies because their MPS uniqueness differs.
function run_instance(objective,n,i,w,strategy,sp,io_lock)
    seed=20261012+100_000*(objective=="quadratic")+100n+i
    coefficients=lock(io_lock) do
        load(input_path(objective,n,i),objective=="linear" ? "c" : "q")
    end
    source_rng=MersenneTwister(seed)
    initial_sha=Ref("");calls=Ref(0)
    function source!(X)
        fill!(X,0)
        for col in axes(X,2), (row,j) in enumerate(randperm(source_rng,n))
            X[(row-1)*n+j,col]=1
        end
        calls[]+=1
        if calls[]==1;initial_sha[]=bytes2hex(sha256(UInt8.(vec(X))));end
        nothing
    end
    objective_function=objective=="linear" ? (x->dot(coefficients,x)) : (x->dot(x,coefficients*x))
    problem=U.OptimizationProblem(A=U.assignment_matrix(n),b=ones(Int,2n),
        cost_function=objective_function,name="train_$(objective)_$(n)_$(i)")
    Random.seed!(seed+10_000_000)
    measured=@timed U.solve(problem,source!,sp,strategy;diversify=false,barrier=false,
        parallel=false,print_stats=false,mixed_proportion=w,rank_weight=w)
    result=measured.value
    all(problem.A*result.incub .== problem.b) || error("Invalid incumbent")
    isfinite(result.c_min) || error("Invalid cost")
    diagnostics=(;seed,initial_training_sha256=initial_sha[],seconds=measured.time,
        allocated_bytes=measured.bytes,gc_seconds=measured.gctime,source_calls=calls[])
    result,diagnostics
end
function run_group(slug,objective,ns,ids,weights,sp,workers,output,code_hash;check=false)
    strategy=STRATEGIES[slug];hashes=Dict{Tuple{Int,Int},String}()
    for n in ns,i in ids
        path=input_path(objective,n,i);isfile(path) || error("Missing training input: $path")
        hashes[(n,i)]=bytes2hex(sha256(read(path)))
    end
    config=(;protocol=1,split="train",strategy,objective,ns,ids,weights,hashes,code_hash,
        parameters=NamedTuple{fieldnames(U.SolverParams)}(Tuple(getfield(sp,k) for k in fieldnames(U.SolverParams))),
        seed_rule="20261012 + 100000*(quadratic) + 100*n + i; model adds 10000000",
        julia_version=string(VERSION),threads=Threads.nthreads(),blas_threads=1)
    @info "Training tuning group" strategy objective instances=length(ns)*length(ids) weights workers output
    check && return
    mkpath(output);path=joinpath(output,"res_$(slug)_$(objective).jld2")
    # Preserve the dictionary layouts used by run_BestCost.jl/run_Strategy.jl.
    baseline=slug=="best_cost"
    result_dict=baseline ? Dict(strategy=>Dict(n=>Dict{Int,U.SolverStatistics}() for n in ns)) :
        Dict(strategy=>Dict(w=>Dict(n=>Dict{Int,U.SolverStatistics}() for n in ns) for w in weights))
    diagnostics=Dict{Tuple{Float64,Int,Int},Any}()
    if isfile(path)
        saved=load(path);saved["configuration"]==config || error("Checkpoint mismatch: $path")
        result_dict=saved["res_dict"];diagnostics=saved["diagnostics"]
    end
    bucket(w,n)=baseline ? result_dict[strategy][n] : result_dict[strategy][w][n]
    jobs=[(w,n,i) for w in weights for n in ns for i in ids if !haskey(bucket(w,n),i)]
    isempty(jobs) && return @info "Already complete" path
    queue=Channel{Tuple{Float64,Int,Int}}(length(jobs))
    foreach(j->put!(queue,j),jobs);close(queue)
    io_lock=ReentrantLock();failures=String[]
    @sync for worker in 1:min(workers,length(jobs))
        Threads.@spawn for (w,n,i) in queue
            try
                result,info=run_instance(objective,n,i,w,strategy,sp,io_lock)
                lock(io_lock) do
                    bucket(w,n)[i]=result;diagnostics[(w,n,i)]=info
                    temp=path*".tmp"
                    jldsave(temp;res_dict=result_dict,solver_params=sp,configuration=config,diagnostics)
                    mv(temp,path;force=true)
                    @info "Saved" strategy objective w n i cost=result.c_min seconds=info.seconds
                end
            catch err
                lock(io_lock) do
                    push!(failures,"w=$w n=$n i=$i: $(sprint(showerror,err))")
                    @error "Instance failed; completed results are retained" strategy objective w n i exception=(err,catch_backtrace())
                end
            end
        end
    end
    isempty(failures) || error(join(failures,"\n"))
end
function main(args=ARGS)
    opt,smoke,check=options(args);BLAS.set_num_threads(1)
    code_hash=implementation_hash();ns=smoke ? [12] : collect(12:16);ids=smoke ? [1] : collect(1:50)
    slug=opt["strategy"];weights=copy(WEIGHT_GRIDS[slug])
    smoke && length(weights)>1 && (weights=weights[[1,end]])
    sp=smoke ? U.SolverParams(NUM_FEASIBLE_SAMPLES=64,NUM_MPS_SAMPLES=128,NUM_GLOBAL_ITER=2,KEEP_NUM_WORST=0.) :
        U.SolverParams(NUM_GLOBAL_ITER=20,KEEP_NUM_WORST=0.,LEARNING_RATE=.05)
    output=isempty(opt["output"]) ? joinpath(ROOT,"data","assignment_add_tuning_v1",smoke ? "smoke_"*code_hash[1:12] : code_hash[1:12]) : abspath(opt["output"])
    objectives=opt["objective"]=="both" ? ["linear","quadratic"] : [opt["objective"]]
    for objective in objectives
        run_group(slug,objective,ns,ids,weights,sp,parse(Int,opt["workers"]),output,code_hash;check)
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
