# Run four independent groups on the existing immutable SA input snapshot.
include("run.jl")
include("penalty.jl")

function run_penalty_worker(;job,root=joinpath(ROOT,"data","sa_cluster_v2"),
        output=joinpath(ROOT,"data","sa_penalty_v1","results"),budgets=[10,50,200],count=40,worker=1,workers=32)
    job in 0:3 || error("job must be 0..3")
    1<=worker<=workers<=32 || error("Invalid worker partition")
    all(>(0),budgets) && allunique(budgets) || error("Invalid budgets")
    family,objective_kind=(("assignment","quadratic"),("assignment","entropic"),("partition","quadratic"),("partition","entropic"))[job+1]
    manifest_path=joinpath(root,"manifest.jld2");manifest=load(manifest_path,"manifest")
    1<=count<=manifest.count || error("Invalid task count")
    entries=sort(filter(r->r.family==family && r.instance<=count,manifest.records);by=r->(r.instance,r.n))
    @assert length(entries)==count*length(manifest.sizes)
    entries=entries[worker:workers:end]
    sources=source_fingerprint()
    for f in ("penalty.jl","run_penalty.jl")
        sources["scripts/sa_cluster/"*f]=bytes2hex(sha256(read(joinpath(@__DIR__,f))))
    end
    protocol=(version=1,family=family,objectives=[objective_kind],sizes=manifest.sizes,
        count=count,worker=worker,workers=workers,assigned=[(e.n,e.instance) for e in entries],
        budgets=budgets,methods=["SA_penalty"],initial_samples=400,sa_chains=10,
        penalty="alpha * sum(abs2, A*x-b)",proposal="one uniform binary bit flip",
        alpha_rule="10 * max(initial cost range, initial cost std, 1e-8*max(1,max(abs(initial costs))))",
        temperature="T0=max(std,alpha*median(nonzero squared column norms)/log(2)); Tend=max(0.001*std,1e-10*scale)",
        timing="initial objective calls, alpha setup, residual evaluation and optimization included; JIT and input loading excluded",
        manifest_sha256=bytes2hex(sha256(read(manifest_path))),sources=sources,
        julia_version=string(VERSION),optim_version=string(pkgversion(Optim)))
    @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
    mkpath(output);path=joinpath(output,"$(family)_$(objective_kind)_worker$(worker).jld2")
    results=Dict{Tuple{Int,Int,String,Int,String},Any}()
    if isfile(path)
        old=load(path);old["protocol"]==protocol || error("Checkpoint protocol mismatch; use another output directory")
        results=old["results"]
    end
    lockpath=path*".lock";ispath(lockpath) && error("Worker already locked: $lockpath")
    mkdir(lockpath)
    hardware=(cpu_model=first(Sys.cpu_info()).model,julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        slurm_job=get(ENV,"SLURM_JOB_ID","local"),slurm_task=get(ENV,"SLURM_ARRAY_TASK_ID","none"),
        slurm_cpu_bind=get(ENV,"SLURM_CPU_BIND_LIST","not recorded"))
    warmed=false
    try
        isempty(entries) && atomic_save(path;results,protocol)
        for entry in entries
            all(haskey(results,(entry.n,entry.instance,objective_kind,b,"SA_penalty")) for b in budgets) && continue
            inputpath=joinpath(root,"inputs",entry.file)
            @assert bytes2hex(sha256(read(inputpath)))==entry.sha256
            record=load(inputpath,"record")
            @assert record.family==family && record.n==entry.n && record.repetition==entry.instance
            @assert all(record.A*record.initial .== record.b)
            objective=make_objective(record,objective_kind)
            if !warmed
                penalty_trial(record,Inf;objective,restarts=1,max_steps_per_chain=100)
                warmed=true
            end
            for budget in budgets
                key=(entry.n,entry.instance,objective_kind,budget,"SA_penalty")
                haskey(results,key) && continue
                GC.gc();timed=@timed penalty_trial(record,Float64(budget);objective)
                r=timed.value
                @assert all(first(h)<=budget for h in r["history"])
                merge!(r,Dict("input_file"=>entry.file,"input_sha256"=>entry.sha256,
                    "allocated_bytes"=>timed.bytes,"gc_seconds"=>timed.gctime,
                    "process_maxrss_bytes"=>Sys.maxrss(),"outer_wall_seconds"=>timed.time,"hardware"=>hardware))
                results[key]=r;atomic_save(path;results,protocol)
                println("$family n=$(entry.n) i=$(entry.instance) $objective_kind $(budget)s: feasible best=$(r["c_min"]), alpha=$(r["alpha"]), calls=$(r["objective_calls"]), feasible calls=$(r["feasible_calls"])");flush(stdout)
            end
        end
    finally
        rm(lockpath)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    options=Dict{String,String}()
    for arg in ARGS
        startswith(arg,"--") && occursin('=',arg) || error("Use --name=value")
        k,v=split(arg[3:end],'=';limit=2)
        k in ("job","worker","workers","root","output","budgets","count") || error("Unknown option $k")
        options[k]=v
    end
    haskey(options,"job") || error("--job required")
    run_penalty_worker(;job=parse(Int,options["job"]),worker=parse(Int,get(options,"worker","1")),
        workers=parse(Int,get(options,"workers","32")),count=parse(Int,get(options,"count","40")),
        budgets=parse.(Int,split(get(options,"budgets","10,50,200"),',')),
        root=abspath(get(options,"root",joinpath(ROOT,"data","sa_cluster_v2"))),
        output=abspath(get(options,"output",joinpath(ROOT,"data","sa_penalty_v1","results"))))
end
