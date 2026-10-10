#!/usr/bin/env julia
# Separate sequential experiment; no production solver or old results modified.
include("run_SetPartitioning.jl")
include("generate_PartitionLocality.jl")
BLAS.set_num_threads(1)

function locality_add_arm(problem,sampler,initial,sp,seed,strategy;external=5000)
    # The training/selection sequence follows src/solver.jl. Differences:
    # exactly `external` fresh draws (not fill-to-20000), explicit per-iteration
    # seeds, read-only diagnostics, and incumbent over ALL evaluated candidates,
    # including the last external batch. Selection/elite retention are unchanged.
    cache=Dict{Tuple,Float64}()
    cost(x)=get!(cache,Tuple(x)) do; Float64(problem.cost_function(x)); end
    T,costs=sort_and_truncate(copy(initial),cost,size(initial,2))
    initial_training_sha256=PartitionEELS.training_fingerprint(T)
    elite_count=max(1,floor(Int,sp.UTILITY_FRACTION*size(T,2)))
    best_samples=T[:,1:elite_count]
    stats=SolverStatistics();history=[Set{Vector{Int}}() for _ in 1:problem.num_vars]
    observations=NamedTuple[];start=time_ns()
    function record!(pool)
        vals=cost.(eachcol(pool));j=argmin(vals)
        if vals[j]<stats.c_min;stats.c_min=vals[j];stats.incub=copy(pool[:,j]);end
        push!(stats.learning_curve,stats.c_min)
        push!(stats.utility_curve,mean(sort(vals)[1:min(elite_count,length(vals))]))
    end
    record!(T)
    encode(v)=foldl(|,(UInt32(v[j])<<(j-1) for j in eachindex(v));init=UInt32(0))
    masks=[encode(@view problem.A[:,j]) for j in 1:problem.num_vars]
    full=encode(problem.b)
    for k in 1:sp.NUM_GLOBAL_ITER
        mps=OhMyU1.build_mps_from_feasible_samples(problem,T,1,false)
        OhMyU1.orthogonalize!(mps)
        support=Float64(OhMyU1.u1_norm(mps)^2);OhMyU1.normalize!(mps)
        temperature=std(costs)
        weights=temperature>0 ? exp.(-(costs.-minimum(costs))./temperature) : ones(length(costs))
        weights./=sum(weights)
        OhMyU1.train_nondeg!(mps,sp.NUM_SWEEP_ITER,T,weights,OhMyU1.TrainParams(sp.LEARNING_RATE,10^4))
        OhMyU1.normalize!(mps)
        Random.seed!(seed+10_000_000k)
        samples=zeros(Int,problem.num_vars,sp.NUM_MPS_SAMPLES)
        OhMyU1.sample_nondeg!(mps,samples,sp.NUM_MPS_SAMPLES)
        @assert all(problem.A*samples .== problem.b)
        model=unique(hcat(samples,best_samples);dims=2)
        model,model_costs=sort_and_truncate(model,cost,size(model,2))
        best_samples=model[:,1:min(elite_count,size(model,2))]
        for j in eachindex(history);union!(history[j],mps.LinkIndices[j+1].Charges);end
        fresh=zeros(Int,problem.num_vars,external)
        fill_partition_vectors!(MersenneTwister(seed+300_000_000+10_000_000k),fresh,sampler)
        @assert all(problem.A*fresh .== problem.b)
        known=[Set(encode(q) for q in h) for h in history]
        distances=Int[]
        for x in eachcol(fresh)
            q=full;distance=0
            for j in eachindex(x)
                x[j]==1 && (q⊻=masks[j]);distance+=!(q in known[j])
            end
            push!(distances,distance)
        end
        buffer=zeros(Int,length(problem.b))
        @assert all(distances[j]==OhMyU1.graph_dist_global!(buffer,@view(fresh[:,j]),mps,history) for j in 1:min(16,external))
        pool=hcat(model,fresh)
        record!(pool)
        stats.num_samples_in_mps=round(Int,support)
        stats.num_unique_samples=size(unique(samples;dims=2),2)
        stats.temperature=temperature
        push!(observations,(;iteration=k,support,training_size=size(T,2),
            model_candidates=size(model,2),external_draws=external,
            external_sha256=PartitionEELS.training_fingerprint(fresh),
            external_zero_fraction=count(iszero,distances)/external,
            external_mean_distance=mean(distances),unique_evaluations=length(cache),
            elapsed_seconds=(time_ns()-start)/1e9))
        T,stats.num_principal_new,stats.max_graph_dist,stats.min_graph_dist,stats.num_to_keep_worst=
            OhMyU1.pick_diverse_data!(mps,pool,history,OhMyU1.graph_dist_global!,strategy,
                cost,sp,false,k;mixed_proportion=.4,rank_weight=.4)
        best_set=Set(eachcol(best_samples))
        keep=findall(j->view(T,:,j) ∉ best_set,axes(T,2))
        T=hcat(best_samples,T[:,keep])
        T,costs=sort_and_truncate(T,cost,size(T,2))
    end
    (;stats,diagnostics=(;initial_training_sha256,observations,
        elapsed_seconds=(time_ns()-start)/1e9,unique_evaluations=length(cache)))
end

function locality_code_hashes()
    files=["scripts/run_PartitionLocality.jl","scripts/run_EELS_optimization.jl",
        "scripts/run_SetPartitioning.jl","src/partition_locality.jl"]
    append!(files,["src/"*f for f in readdir(joinpath(PARTITION_ROOT,"src")) if endswith(f,".jl") && !(f in ("deprecated.jl","future.jl","py_tools.jl"))])
    Dict(f=>bytes2hex(sha256(read(joinpath(PARTITION_ROOT,f)))) for f in unique(files))
end

function run_locality(span;smoke=false,instances=1:40)
    span in LOCALITY_SPANS || error("Unknown span")
    Threads.nthreads()==1 || error("Run with --threads=1; parallelism is across isolated Slurm jobs")
    sp=SolverParams(NUM_FEASIBLE_SAMPLES=smoke ? 40 : 500,NUM_MPS_SAMPLES=smoke ? 100 : 5000,
        NUM_GLOBAL_ITER=smoke ? 2 : 15,NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,
        LEARNING_RATE=.05,UTILITY_FRACTION=.05,KEEP_NUM_WORST=0.)
    external=smoke ? 100 : 5000
    root=joinpath(LOCALITY_ROOT,smoke ? "smoke" : "results")
    mkpath(root);lockdir=joinpath(root,".running_span$span")
    mkdir(lockdir) # atomic refusal of a second writer, including direct Julia runs
    try
        hashes=locality_code_hashes()
        for i in instances
            input=locality_path(span,i);d=load(input);p=d["instance"]
            @assert (d["specification"].span,d["specification"].i)==(span,i)
            sampler=ExactPartitionSampler(p.tiles,length(p.b))
            @assert sampler.counts[UInt128(0)]==d["structural"].total
            initial=Int.(d["initial"][:,1:sp.NUM_FEASIBLE_SAMPLES]);@assert all(p.A*initial .== p.b)
            rawhash=PartitionEELS.training_fingerprint(initial)
            problem=OptimizationProblem(A=p.A,b=p.b,cost_function=partition_cost(d["objectives"],"entropic",length(p.b)),name="locality")
            seed=d["specification"].seed+100_000_000
            for mode in ("EELS","ADD")
                path=joinpath(root,"res_$(mode)_entropic_span$(span).jld2")
                arm=mode=="EELS" ? "EELS" : PARTITION_ADD_STRATEGY
                config=(;protocol=1,span,mode,external=mode=="EELS" ? 0 : external,
                    parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(sp,f) for f in fieldnames(SolverParams))),
                    code_hashes=hashes,gamma=1,filter_weight=.4,julia_version=string(VERSION),
                    incumbent="best of all evaluated candidates including final external batch")
                res_dict=Dict{String,Any}("best_cost"=>Dict(15=>Dict{Int,SolverStatistics}()),
                    arm=>(mode=="EELS" ? Dict(15=>Dict{Int,SolverStatistics}()) : Dict(.4=>Dict(15=>Dict{Int,SolverStatistics}()))))
                diagnostics=Dict{Int,Any}()
                if isfile(path)
                    saved=load(path);saved["configuration"]==config || error("Checkpoint/code mismatch")
                    res_dict=saved["res_dict"];diagnostics=saved["diagnostics"]
                end
                input_sha=bytes2hex(sha256(read(input)))
                if haskey(diagnostics,i)
                    diagnostics[i].input_sha256==input_sha || error("Input changed")
                    continue
                end
                pair=Dict{String,Any}()
                for strategy in (isodd(i) ? ["best_cost",arm] : [arm,"best_cost"])
                    println("span=$span i=$i mode=$mode arm=$strategy");flush(stdout)
                    if mode=="EELS"
                        stats,diag=PartitionEELS.run_arm(problem,initial,sp,seed;eels=strategy==arm)
                        pair[strategy]=(;stats,diagnostics=diag)
                    else
                        pair[strategy]=locality_add_arm(problem,sampler,initial,sp,seed,strategy;external)
                    end
                end
                @assert pair["best_cost"].diagnostics.initial_training_sha256==pair[arm].diagnostics.initial_training_sha256
                if mode=="ADD"
                    @assert getproperty.(pair["best_cost"].diagnostics.observations,:external_sha256)==getproperty.(pair[arm].diagnostics.observations,:external_sha256)
                end
                @assert PartitionEELS.training_fingerprint(initial)==rawhash
                for v in values(pair)
                    @assert all(p.A*v.stats.incub .== p.b)
                    @assert isapprox(problem.cost_function(v.stats.incub),v.stats.c_min;atol=1e-10,rtol=1e-10)
                    @assert length(v.stats.learning_curve)==sp.NUM_GLOBAL_ITER+1
                    @assert all(diff(v.stats.learning_curve).<=1e-12)
                end
                res_dict["best_cost"][15][i]=pair["best_cost"].stats
                branch=mode=="EELS" ? res_dict[arm] : res_dict[arm][.4]
                branch[15][i]=pair[arm].stats
                diagnostics[i]=(;input_sha256=input_sha,initial_sha256=rawhash,seed,
                    structural=d["structural"],cpu=Sys.CPU_NAME,threads=Threads.nthreads(),
                    blas_threads=BLAS.get_num_threads(),arms=Dict(k=>v.diagnostics for (k,v) in pair))
                jldsave(path*".tmp";res_dict,solver_params=sp,configuration=config,diagnostics)
                Base.Filesystem.rename(path*".tmp",path)
                GC.gc()
            end
        end
    finally
        rm(lockdir)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    if ARGS==["--smoke"]
        run_locality(5;smoke=true,instances=1:1)
        run_locality(18;smoke=true,instances=1:1)
    elseif length(ARGS)==2 && ARGS[1]=="--span"
        run_locality(parse(Int,ARGS[2]))
    else
        error("Usage: --smoke | --span 5 (or 6,8,10,12,15,18)")
    end
end
