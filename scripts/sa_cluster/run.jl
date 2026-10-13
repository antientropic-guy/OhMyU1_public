# Independent single-core shards within four family/objective allocations.
include("prepare.jl")

function make_objective(record, kind)
    kind=="quadratic" && return BinaryQuadratic(record.Q,Int[])
    kind=="entropic" || error("Unknown objective")
    o=record.entropic
    return function(x)
        z=(o.scenarios*x)./(o.tau*o.scale)
        peak=maximum(z)
        dot(o.c,x)/o.scale+o.tau*(peak+log(sum(exp(v-peak) for v in z))-log(length(z)))
    end
end

function adapter_context(record)
    input=(n=record.n,repetition=record.repetition,seed=record.seed,initial=record.initial,Q=record.Q)
    if record.family=="partition"
        p=(A=record.A,b=record.b,tiles=record.tiles,n=record.n)
        masks,patches=patch_catalogue(p)
        validate_patches(p,record.initial,masks,patches)
        neighbor=(x,y,rng)->patch_neighbor!(x,y,rng,masks,patches)
        factory=()->partition_indicator(p,record.initial,parse(BigInt,record.support))
    else
        neighbor=nothing;factory=nothing
    end
    (;input,neighbor,factory)
end

function source_fingerprint()
    files=vcat(["Project.toml","Manifest.toml"],
        [joinpath("src",f) for f in sort(readdir(joinpath(ROOT,"src"))) if endswith(f,".jl")],
        [joinpath("scripts","sa_cluster",f) for f in ("adapters.jl","structural.jl","run.jl","prepare.jl","Project.toml","Manifest.toml")])
    Dict(replace(f,'\\'=>'/')=>bytes2hex(sha256(read(joinpath(ROOT,f)))) for f in files if isfile(joinpath(ROOT,f)))
end

function run_worker(;job,root=joinpath(ROOT,"data","sa_cluster_v2"),output=joinpath(root,"results"),
                    budgets=[10,50,200],count=40,worker=1,workers=32)
    job in 0:3 || error("job must be in 0:3")
    all(>(0),budgets) && allunique(budgets) || error("Positive unique budgets required")
    1<=worker<=workers<=32 || error("Require 1 <= worker <= workers <= 32")
    family,objective_kind=(("assignment","quadratic"),("assignment","entropic"),("partition","quadratic"),("partition","entropic"))[job+1]
    manifest_path=joinpath(root,"manifest.jld2");manifest=load(manifest_path,"manifest")
    1<=count<=manifest.count || error("Requested count exceeds frozen input snapshot")
    records=sort(filter(r->r.family==family && r.instance<=count,manifest.records);by=r->(r.instance,r.n))
    length(records)==count*length(manifest.sizes) || error("Incomplete input manifest")
    records=records[worker:workers:end]
    Threads.nthreads()==1 && BLAS.get_num_threads()==1 || error("Use one Julia and BLAS thread")
    mkpath(output)
    path=joinpath(output,"$(family)_$(objective_kind)_worker$(worker).jld2")
    protocol=(version=2,family=family,sizes=manifest.sizes,worker=worker,workers=workers,
        assigned=[(r.n,r.instance) for r in records],count=count,budgets=budgets,methods=["SA","U1_data_EELS","U1_full"],
        objectives=[objective_kind],initial_samples=400,mps_samples=10000,elite_count=400,
        lr=0.05,sweeps=1,backend="legacy",charge_degeneracy=1,sa_chains=10,
        mps_reset="every iteration",elite_selection="best distinct old elites plus new samples",
        mps_temperature="std of current elite costs",sa_temperature="initial std to 0.001 over each chain",
        timing="model construction included; SA move catalogue and compilation excluded",
        manifest_sha256=bytes2hex(sha256(read(manifest_path))),sources=source_fingerprint(),
        julia_version=string(VERSION),optim_version=string(pkgversion(Optim)))
    results=Dict{Tuple{Int,Int,String,Int,String},Any}()
    if isfile(path)
        previous=load(path)
        previous["protocol"]==protocol || error("Existing output has a different protocol/input/code; choose a new output directory")
        results=previous["results"]
    end
    # Atomic mkdir prevents two submissions from writing the same shard.
    lockpath=path*".lock"
    isdir(lockpath) && error("Worker lock exists: $lockpath. Remove only after confirming the prior worker has ended.")
    mkdir(lockpath)
    hardware=(cpu=Sys.CPU_NAME,cpu_model=first(Sys.cpu_info()).model,architecture=Sys.MACHINE,os=string(Sys.KERNEL),
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        slurm_job=get(ENV,"SLURM_JOB_ID","local"),slurm_task=get(ENV,"SLURM_ARRAY_TASK_ID","none"),
        slurm_cpu_bind=get(ENV,"SLURM_CPU_BIND_LIST","not recorded"))
    warmed=Set{String}()
    try
        # Empty shards are valid for reduced smoke designs and remain collectable.
        isempty(records) && atomic_save(path;results,protocol)
        for entry in records
            n=entry.n
            inputpath=joinpath(root,"inputs",entry.file)
            bytes2hex(sha256(read(inputpath)))==entry.sha256 || error("Input checksum mismatch")
            record=load(inputpath,"record")
            @assert record.family==family && record.n==n && record.repetition==entry.instance
            @assert all(record.A*record.initial .== record.b)
            ctx=adapter_context(record)
            # Rotate objective and method order without outcome-dependent choices.
            for kind in circshift(protocol.objectives,mod(entry.instance-1,2))
                all(haskey(results,(n,entry.instance,kind,b,m)) for b in budgets for m in protocol.methods) && continue
                objective=make_objective(record,kind)
                if !(kind in warmed)
                    sa_trial(ctx.input,Inf;objective,A=record.A,neighbor=ctx.neighbor,restarts=1,max_steps_per_chain=100)
                    for full in (false,true)
                        u1_trial(ctx.input,Inf;objective,A=record.A,full_factory=ctx.factory,full,
                            maxiter=1,samples_per_iteration=100)
                    end
                    push!(warmed,kind)
                end
                for budget in budgets,method in circshift(protocol.methods,mod(entry.instance-1,3))
                    key=(n,entry.instance,kind,budget,method)
                    haskey(results,key) && continue
                    GC.gc()
                    measured=@timed (method=="SA" ?
                        sa_trial(ctx.input,Float64(budget);objective,A=record.A,neighbor=ctx.neighbor) :
                        u1_trial(ctx.input,Float64(budget);objective,A=record.A,full_factory=ctx.factory,full=method=="U1_full"))
                    r=measured.value
                    @assert all(record.A*r["solution"] .== record.b)
                    @assert all(x->x in (0,1),r["solution"])
                    @assert isapprox(objective(r["solution"]),r["c_min"];atol=1e-9,rtol=1e-12)
                    @assert all(first(h)<=budget for h in r["history"])
                    @assert all(diff(last.(r["history"])).<=0)
                    @assert all(first(e)<=r["objective_calls"] for e in r["evaluation_history"])
                    merge!(r,Dict("input_file"=>entry.file,"input_sha256"=>entry.sha256,
                        "allocated_bytes"=>measured.bytes,"gc_seconds"=>measured.gctime,
                        "process_maxrss_bytes"=>Sys.maxrss(),"outer_wall_seconds"=>measured.time,
                        "zero_completed_iterations"=>r["completed_iterations"]==0,
                        "hardware"=>hardware))
                    results[key]=r
                    atomic_save(path;results,protocol)
                    println("$family n=$n i=$(entry.instance) $kind budget=$budget $method: $(r["c_min"]), calls=$(r["objective_calls"]), elapsed=$(r["elapsed_seconds"])");flush(stdout)
                end
            end
        end
    finally
        rm(lockpath)
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    opts=Dict{String,String}()
    for arg in ARGS
        startswith(arg,"--") && occursin('=',arg) || error("Use --name=value arguments")
        k,v=split(arg[3:end],'=';limit=2);k in ("job","root","output","budgets","count","worker","workers") || error("Unknown option: $k")
        opts[k]=v
    end
    haskey(opts,"job") || error("Required: --job=0|1|2|3")
    root=abspath(get(opts,"root",joinpath(ROOT,"data","sa_cluster_v2")))
    run_worker(;job=parse(Int,opts["job"]),root,output=abspath(get(opts,"output",joinpath(root,"results"))),
        budgets=parse.(Int,split(get(opts,"budgets","10,50,200"),',')),count=parse(Int,get(opts,"count","40")),
        worker=parse(Int,get(opts,"worker","1")),workers=parse(Int,get(opts,"workers","32")))
end
