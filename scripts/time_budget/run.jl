#!/usr/bin/env julia
# Isolated-process, elapsed-time comparisons. No production source is modified.
using OhMyU1, Random, LinearAlgebra, Statistics, JLD2, SHA, Dates
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const U=OhMyU1
module Protocols
include("../run_PartitionLocality.jl")
end
const E=Protocols.PartitionEELS
BLAS.set_num_threads(1)

struct BudgetExpired <: Exception end
mutable struct Recorder
    start::UInt64
    limit::Float64
    calls::Int
    best::Float64
    incumbent::Vector{Int}
    improvements::Vector{NamedTuple}
    iterations::Vector{NamedTuple}
    iteration::Int
    call_trace::Vector{NamedTuple}
    budgets::Vector{Float64}
    completed_calls::Int
    budget_calls::Dict{Float64,Int}
end
elapsed(r)=(time_ns()-r.start)/1e9
function capture_calls(r,t)
    for b in r.budgets
        if t>=b && !haskey(r.budget_calls,b)
            r.budget_calls[b]=r.completed_calls
        end
    end
end
function gate(r,iteration=r.iteration)
    r.iteration=iteration
    t=elapsed(r)
    capture_calls(r,t)
    if isempty(r.call_trace) || t-last(r.call_trace).seconds>=0.1
        push!(r.call_trace,(seconds=t,calls=r.calls))
    end
    t>=r.limit && throw(BudgetExpired())
end
function counted(r,raw)
    function objective(x)
        gate(r)
        r.calls+=1
        value=Float64(raw(x))
        isfinite(value) || error("Non-finite objective")
        t=elapsed(r)
        capture_calls(r,t)
        r.completed_calls+=1
        # An objective that finishes after the deadline cannot improve the result.
        t>=r.limit && throw(BudgetExpired())
        if value<r.best
            r.best=value;r.incumbent=collect(x)
            push!(r.improvements,(seconds=t,calls=r.calls,cost=value,iteration=r.iteration))
        end
        value
    end
end
const ACTIVE=Ref{Recorder}()
function checkpoint(k)
    r=ACTIVE[]
    gate(r,k)
    push!(r.iterations,(iteration=k,seconds=elapsed(r),calls=r.calls,cost=r.best))
end

function replace_required(s,a,b)
    occursin(a,s) || error("Protocol source changed: missing $(repr(a))")
    replace(s,a=>b;count=1)
end
function install()
    # Keep the production ADD algorithm, inserting only safe-point deadline checks.
    source=replace(read(joinpath(ROOT,"src","solver.jl"),String),"\r\n"=>"\n")
    source=source[findfirst("function solve(",source).start:end]
    source=replace_required(source,"function solve(","function timed_add_solve(")
    source=replace_required(source,"for k in 1:sp.NUM_GLOBAL_ITER",
        "for k in 1:sp.NUM_GLOBAL_ITER\n        TIME_GATE[](k-1)")
    source=replace_required(source,"        # Sampling","        TIME_GATE[](k-1)\n        # Sampling")
    source=replace_required(source,"        # Keep best samples","        TIME_GATE[](k-1)\n        # Keep best samples")
    source=replace_required(source,"        if print_stats\n            print_statistics(solver_stats)",
        "        TIME_CHECKPOINT[](k)\n        if print_stats\n            print_statistics(solver_stats)")
    Base.include_string(U,"const TIME_GATE=Ref{Function}()\nconst TIME_CHECKPOINT=Ref{Function}()\n"*source)

    # Reuse the published locality protocol, excluding expensive analysis-only
    # graph diagnostics that would otherwise penalize even the Best Cost arm.
    s=replace(read(joinpath(ROOT,"scripts","run_PartitionLocality.jl"),String),"\r\n"=>"\n")
    s=s[findfirst("function locality_add_arm",s).start:prevind(s,findfirst("function locality_code_hashes",s).start)]
    s=replace_required(s,"function locality_add_arm","function timed_locality_arm")
    a=findfirst("    encode(v)=",s).start;b=findfirst("    for k in",s).start
    s=s[1:prevind(s,a)]*s[b:end]
    a=findfirst("        known=",s).start;b=findfirst("        pool=hcat",s).start
    s=s[1:prevind(s,a)]*s[b:end]
    a=findfirst("        push!(observations,",s).start;b=findfirst("        T,stats.num_principal_new",s).start
    s=s[1:prevind(s,a)]*s[b:end]
    s=replace_required(s,"for k in 1:sp.NUM_GLOBAL_ITER","for k in 1:sp.NUM_GLOBAL_ITER\n        TIME_GATE[](k-1)")
    s=replace_required(s,"        Random.seed!(seed+10_000_000k)","        TIME_GATE[](k-1)\n        Random.seed!(seed+10_000_000k)")
    s=replace_required(s,"        T,costs=sort_and_truncate(T,cost,size(T,2))",
        "        T,costs=sort_and_truncate(T,cost,size(T,2))\n        TIME_CHECKPOINT[](k)")
    Base.include_string(Protocols,"const TIME_GATE=Ref{Function}()\nconst TIME_CHECKPOINT=Ref{Function}()\n"*s)
end
install()
U.TIME_GATE[]=k->gate(ACTIVE[],k)
U.TIME_CHECKPOINT[]=checkpoint
Protocols.TIME_GATE[]=k->gate(ACTIVE[],k)
Protocols.TIME_CHECKPOINT[]=checkpoint

function tasks()
    result=NamedTuple[]
    # Interleave comparison types to reduce the tail of the allocation.
    for i in 1:50
        for n in 12:16,o in ("linear","quadratic"),mode in ("add","eels")
            path=E.instance_path("test",o,n,i)
            push!(result,(id="$(mode)_$(o)_n$(n)_i$i",mode,objective=o,n,i,w=0,m=2n,path))
        end
        if i<=15
            for m in 18:20,n in 13:15,w in Protocols.LOCALITY_SPANS
                push!(result,(id="transfer_w$(w)_m$(m)_n$(n)_i$i",mode="transfer",
                    objective="entropic",n,i,w,m,path=Protocols.grid_path(w,m,n,i)))
            end
        end
    end
    @assert length(result)==1945 && allunique(getproperty.(result,:id))
    result
end
function codehash()
    paths=filter(p->endswith(p,".jl"),readdir(joinpath(ROOT,"src");join=true))
    append!(paths,[joinpath(ROOT,"scripts",f) for f in
        ("run_PartitionLocality.jl","run_SetPartitioning.jl","run_EELS_optimization.jl",
         "generate_set_partitioning_instances.jl","generate_PartitionLocality.jl","generate_PartitionLocalityGrid.jl")])
    append!(paths,[@__FILE__,joinpath(@__DIR__,"monitor.py"),joinpath(ROOT,"scripts/bash/run_time_budget.sh")])
    append!(paths,[joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")])
    bytes2hex(sha256(join(bytes2hex(sha256(read(p))) for p in sort(paths))))
end
function assignment_source(n,seed,initial)
    batch=Ref(0)
    function source!(X)
        batch[]+=1
        if batch[]==1
            copyto!(X,initial)
        else
            rng=MersenneTwister(seed+300_000_000+10_000_000*(batch[]-1))
            fill!(X,0)
            for col in axes(X,2),(row,j) in enumerate(randperm(rng,n))
                X[(row-1)*n+j,col]=1
            end
        end
    end
end
function inputs(t;small=false)
    if t.mode=="transfer"
        d=load(t.path);p=d["instance"]
        sampler=Protocols.ExactPartitionSampler(p.tiles,length(p.b))
        @assert sampler.counts[UInt128(0)]==d["structural"].total
        X=Int.(d["initial"][:,1:(small ? 24 : 500)])
        raw=Protocols.partition_cost(d["objectives"],"entropic",length(p.b))
        return (;A=p.A,b=p.b,initial=X,raw,sampler,seed=d["specification"].seed+100_000_000,
            structural=d["structural"])
    end
    n=small ? 4 : t.n
    seed=20261022+100_000*(t.objective=="quadratic")+100t.n+t.i
    rng=MersenneTwister(seed)
    count=small ? 24 : t.mode=="eels" ? 400 : 1000
    X=zeros(Int,n*n,count)
    for col in axes(X,2),(row,j) in enumerate(randperm(rng,n))
        X[(row-1)*n+j,col]=1
    end
    c=small ? (t.objective=="linear" ? randn(rng,n*n) : randn(rng,n*n,n*n)) :
        load(t.path,t.objective=="linear" ? "c" : "q")
    raw=t.objective=="linear" ? (x->dot(c,x)) : (x->dot(x,c*x))
    (;A=U.assignment_matrix(n),b=ones(Int,2n),initial=X,raw,sampler=n,seed,structural=nothing)
end
function execute(t,d,arm,limit;small=false,maxiter=typemax(Int),budgets=Float64[limit])
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=size(d.initial,2),
        NUM_MPS_SAMPLES=small ? 48 : t.mode=="transfer" ? 5000 : 10000,
        NUM_GLOBAL_ITER=maxiter,NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,
        LEARNING_RATE=.05,UTILITY_FRACTION=.05,KEEP_NUM_WORST=0.)
    r=Recorder(time_ns(),limit,0,Inf,Int[],NamedTuple[],NamedTuple[],0,NamedTuple[],
        budgets,0,Dict{Float64,Int}())
    ACTIVE[]=r
    f=counted(r,d.raw)
    p=U.OptimizationProblem(A=d.A,b=d.b,cost_function=f)
    initial_evaluations=t.mode=="add" ? size(d.initial,2) : size(unique(d.initial;dims=2),2)
    source=t.mode=="add" ? assignment_source(d.sampler,d.seed,d.initial) : nothing
    observer=function(event,k,mps,T)
        gate(r,k-1)
        event==:after_training && checkpoint(k)
    end
    Random.seed!(d.seed+10_000_000)
    GC.gc()
    unix_start=time();r.start=time_ns()
    measured=@timed try
        if t.mode=="add"
            U.timed_add_solve(p,source,sp,arm=="gdf" ? "best nonzero mixed" : "best_cost";
                diversify=false,barrier=false,parallel=false,print_stats=false,
                mixed_proportion=.4,rank_weight=.4)
        elseif t.mode=="transfer"
            Protocols.timed_locality_arm(p,d.sampler,d.initial,sp,d.seed,
                arm=="gdf" ? "best nonzero mixed" : "best_cost";external=small ? 48 : 5000)
        else
            E.run_persistent_arm(p,d.initial,sp,d.seed;eels=arm=="eels",observer)
        end
        :iteration_limit
    catch err
        err isa BudgetExpired || rethrow()
        :deadline
    end
    actual=elapsed(r);unix_finish=time()
    capture_calls(r,actual)
    isempty(r.incumbent) || @assert all(d.A*r.incumbent .== d.b)
    initial_improvements=filter(x->x.calls<=initial_evaluations,r.improvements)
    initial_cost=get(r.budget_calls,limit,r.completed_calls)>=initial_evaluations && !isempty(initial_improvements) ?
        last(initial_improvements).cost : NaN
    (;recorder=r,parameters=sp,seconds=actual,unix_start,unix_finish,
        initial_cost,initial_evaluations,
        overshoot=max(0.,actual-limit),allocated_bytes=measured.bytes,
        gc_seconds=measured.gctime,termination=measured.value)
end
function csvwrite(path,rows)
    isempty(rows) && return
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows;println(io,join(values(r),','));end
    end
end
function pair_unlocked(t,out,budgets,signature,inputhash;smoke=false)
    d=inputs(t;small=smoke)
    @assert all(d.A*d.initial .== d.b)
    initial_hash=E.training_fingerprint(d.initial)
    candidate=t.mode=="eels" ? "eels" : "gdf"
    arms=isodd(t.i) ? ("best_cost",candidate) : (candidate,"best_cost")
    for arm in arms
        folder=joinpath(out,t.id,arm);file=joinpath(folder,"result.jld2")
        if isfile(file)
            saved=load(file,"configuration")
            @assert saved.signature==signature && saved.inputhash==inputhash && saved.budgets==budgets
            continue
        end
        mkpath(folder)
        println("START $(t.id) $arm");flush(stdout)
        result=execute(t,d,arm,maximum(budgets);small=smoke,budgets)
        @assert E.training_fingerprint(d.initial)==initial_hash
        r=result.recorder
        checkpoints=map(budgets) do b
            seen=filter(x->x.seconds<=b,r.improvements)
            (;budget_s=b,cost=isempty(seen) ? Inf : last(seen).cost,
                objective_calls=get(r.budget_calls,b,r.completed_calls),
                improvement_calls=isempty(seen) ? 0 : last(seen).calls)
        end
        configuration=(;signature,inputhash,budgets,initial_hash,task=t,arm,
            seed=d.seed,julia=string(VERSION),threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
            sampling_backend=t.mode=="eels" ? "legacy" : "table",
            objective_cache=t.mode!="add",filter_weight=.4,initial_generation_timed=false,
            pid=getpid(),host=get(ENV,"HOSTNAME","unknown"),
            monitor_file=get(ENV,"TIME_BUDGET_MONITOR",""))
        # Existing readers can access SolverStatistics; time-resolved data are separate.
        stats=U.SolverStatistics();stats.c_min=r.best;stats.incub=r.incumbent
        stats.learning_curve=Float64[x.cost for x in r.improvements]
        res_dict=Dict(arm=>Dict(t.n=>Dict(t.i=>stats)))
        jldsave(file*".tmp";configuration,result,checkpoints,res_dict,structural=d.structural)
        mv(file*".tmp",file;force=true)
        csvwrite(joinpath(folder,"improvements.csv"),r.improvements)
        csvwrite(joinpath(folder,"iterations.csv"),r.iterations)
        csvwrite(joinpath(folder,"objective_calls.csv"),r.call_trace)
        csvwrite(joinpath(folder,"checkpoints.csv"),checkpoints)
        println("DONE $(t.id) $arm time=$(result.seconds) overshoot=$(result.overshoot) cost=$(r.best)");flush(stdout)
    end
end
function pair(t,out,budgets,signature,inputhash;smoke=false)
    dir=joinpath(out,t.id);mkpath(dir)
    lockdir=joinpath(dir,".running")
    mkdir(lockdir) # Atomic refusal of duplicate writers, including duplicate submissions.
    try
        pair_unlocked(t,out,budgets,signature,inputhash;smoke)
    finally
        rm(lockdir)
    end
end
function main(args=ARGS)
    opt=Dict("job"=>"0","worker"=>"0","workers"=>"32","budgets"=>"10,50,200","output"=>"data/time_budget_v1")
    flags=Set{String}()
    for a in args
        if a in ("--prepare","--smoke");push!(flags,a)
        else
            k,v=split(a[3:end],'=';limit=2);haskey(opt,k) || error("Unknown argument $a");opt[k]=v
        end
    end
    budgets=sort(unique(parse.(Float64,split(opt["budgets"],','))))
    all(x->isfinite(x)&&x>0,budgets) || error("Positive finite budgets required")
    smoke="--smoke" in flags
    out=abspath(joinpath(ROOT,opt["output"]));mkpath(out)
    signature=codehash();manifest=joinpath(out,"manifest.jld2")
    tt=tasks()
    if "--prepare" in flags
        hashes=Dict(t.path=>bytes2hex(sha256(read(t.path))) for t in tt)
        frozen=(;signature,budgets,tasks=tt,input_hashes=hashes)
        if isfile(manifest);@assert load(manifest,"frozen")==frozen "Use a new output directory for changed code/data/budgets"
        else;jldsave(manifest;frozen);end
        println("Validated $(length(tt)) pairs, $(2length(tt)) arms. No optimization performed.")
        return
    end
    @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
    if smoke
        out=joinpath(out,"smoke");mkpath(out)
        chosen=[first(filter(t->t.mode==m,tt)) for m in ("add","eels","transfer")]
        for t in chosen
            d=inputs(t;small=true)
            for arm in ("best_cost",t.mode=="eels" ? "eels" : "gdf")
                execute(t,d,arm,Inf;small=true,maxiter=1)
            end
            pair(t,out,budgets,signature,bytes2hex(sha256(read(t.path)));smoke=true)
        end
        return
    end
    frozen=load(manifest,"frozen")
    @assert frozen.signature==signature && frozen.budgets==budgets
    job=parse(Int,opt["job"]);worker=parse(Int,opt["worker"]);workers=parse(Int,opt["workers"])
    @assert job in 0:3 && 0<=worker<workers<=32
    chosen=tt[job*workers+worker+1:4workers:end]
    # Compile the exact argument types and both arms before any measured execution.
    for mode in ("add","eels","transfer"),objective in (mode=="transfer" ? ("entropic",) : ("linear","quadratic",))
        t=first(filter(t->t.mode==mode && t.objective==objective,tt))
        d=inputs(t;small=true)
        for arm in ("best_cost",mode=="eels" ? "eels" : "gdf")
            execute(t,d,arm,Inf;small=true,maxiter=1)
        end
    end
    for t in chosen
        @assert bytes2hex(sha256(read(t.path)))==frozen.input_hashes[t.path]
        pair(t,out,budgets,signature,frozen.input_hashes[t.path])
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
