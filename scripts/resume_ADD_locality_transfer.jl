#!/usr/bin/env julia
# Validate and migrate completed pairs after removing redundant Pkg.activate calls.
# No optimization, package activation, lock deletion, or source-result mutation.
include("run_ADD_locality_transfer.jl")

const RESUME_OLD_EELS_HASH = "2c8aa3533bec82fbbdeab0ed9fdc9f42a874b00bf13f39d4289ea0d167be24e3"
const RESUME_NEW_EELS_HASH = "0e31c3ecdaf28861273cc1f0517bf484c01b0485416fb70b783f67440bed4c87"
const RESUME_NEW_GENERATOR_HASH = "2cc34bb18cd18afa372619309b73065ab544201c9d3d0b3dbedf8d416f7d5213"

transfer_signature(hashes) = bytes2hex(sha256(join(k*":"*hashes[k] for k in sort(collect(keys(hashes))))))[1:16]

function validate_transfer_checkpoint(path, hashes, inputs)
    d=load(path);c=d["configuration"];sp=d["solver_params"]
    expected_sp=SolverParams(NUM_FEASIBLE_SAMPLES=500,NUM_MPS_SAMPLES=5000,
        NUM_GLOBAL_ITER=15,NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,
        LEARNING_RATE=.05,UTILITY_FRACTION=.05,KEEP_NUM_WORST=0.)
    parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(expected_sp,f) for f in fieldnames(SolverParams)))
    @assert c.parameters==parameters
    @assert all(getfield(sp,f)==getfield(expected_sp,f) for f in fieldnames(SolverParams))
    expected_config=(;protocol=3,span=c.span,elements=c.elements,n=c.n,objective="entropic",
        grid=true,mode="ADD",external=5000,parameters,code_hashes=hashes,gamma=1,
        filter_weight=.4,sampling_backend="table",julia_version=string(VERSION),
        incumbent="best of all evaluated candidates including final external batch")
    @assert c==expected_config "Configuration or code mismatch: $path"
    @assert length(d["diagnostics"])==1
    i=only(keys(d["diagnostics"]));key=(c.span,c.elements,c.n,i)
    @assert haskey(inputs,key)
    @assert basename(path)=="res_ADD_entropic_span$(c.span)_m$(c.elements)_n$(c.n)_i$i.jld2"
    input=grid_path(key...);input_hash=bytes2hex(sha256(read(input)))
    diag=d["diagnostics"][i]
    @assert input_hash==inputs[key]==diag.input_sha256
    source=load(input);p=source["instance"]
    initial=Int.(source["initial"][:,1:500])
    @assert size(p.A)==(c.elements,c.n^2) && all(p.A*initial .== p.b)
    @assert PartitionEELS.training_fingerprint(initial)==diag.initial_sha256
    @assert diag.structural==source["structural"]
    @assert diag.seed==source["specification"].seed+100_000_000
    objective=partition_cost(source["objectives"],"entropic",c.elements)
    training,_=sort_and_truncate(initial,objective,500)
    fingerprint=PartitionEELS.training_fingerprint(training)
    results=d["res_dict"];arm="best nonzero mixed"
    @assert Set(keys(results))==Set(["best_cost",arm])
    @assert Set(keys(results[arm]))==Set([.4])
    for (name,branch) in (("best_cost",results["best_cost"]),(arm,results[arm][.4]))
        @assert Set(keys(branch))==Set([c.n]) && Set(keys(branch[c.n]))==Set([i])
        r=branch[c.n][i];a=diag.arms[name]
        @assert a.initial_training_sha256==fingerprint
        @assert length(r.learning_curve)==16 && all(isfinite,r.learning_curve)
        @assert length(r.utility_curve)==16 && all(isfinite,r.utility_curve)
        @assert all(diff(r.learning_curve).<=1e-12) && last(r.learning_curve)==r.c_min
        @assert isapprox(first(r.learning_curve),minimum(objective(x) for x in eachcol(initial));atol=1e-10,rtol=1e-10)
        @assert length(r.incub)==c.n^2 && all(x->x in (0,1),r.incub) && p.A*r.incub==p.b
        @assert isapprox(objective(r.incub),r.c_min;atol=1e-10,rtol=1e-10)
        @assert length(a.observations)==15
        @assert [o.iteration for o in a.observations]==collect(1:15)
        @assert all(o.external_draws==5000 for o in a.observations)
    end
    @assert getproperty.(diag.arms["best_cost"].observations,:external_sha256)==getproperty.(diag.arms[arm].observations,:external_sha256)
    d
end

function resume_transfer(signature, expected_count; root=joinpath(LOCALITY_GRID_ROOT,"results_add_transfer_v3"))
    occursin(r"^[0-9a-f]{16}$",signature) || error("Expected a 16-character source hash")
    1<=expected_count<=945 || error("Expected count must be between 1 and 945")
    hashes=merge(locality_code_hashes(),Dict("scripts/run_ADD_locality_transfer.jl"=>bytes2hex(sha256(read(joinpath(@__DIR__,"run_ADD_locality_transfer.jl"))))))
    @assert hashes["scripts/run_EELS_optimization.jl"]==RESUME_NEW_EELS_HASH "Recovery only supports the activation-only patch"
    @assert bytes2hex(sha256(read(joinpath(@__DIR__,"generate_set_partitioning_instances.jl"))))==RESUME_NEW_GENERATOR_HASH
    old_hashes=copy(hashes);old_hashes["scripts/run_EELS_optimization.jl"]=RESUME_OLD_EELS_HASH
    @assert transfer_signature(old_hashes)==signature "Other implementation changes detected; refusing to reuse results"
    source=joinpath(root,signature,"entropic")
    target=joinpath(root,transfer_signature(hashes),"entropic")
    @assert source!=target && isdir(source)
    files=sort(filter(p->startswith(basename(p),"res_ADD_entropic_") && endswith(p,".jld2"),readdir(source;join=true)))
    length(files)==expected_count || error("Found $(length(files)) completed pairs, expected $expected_count; no jobs submitted")
    inputs=Dict((r.w,r.m,r.n,r.i)=>r.sha256 for r in load(joinpath(LOCALITY_GRID_ROOT,"manifest.jld2"),"rows"))
    @assert Set(keys(inputs))==Set((t.w,t.m,t.n,t.i) for t in transfer_tasks())
    digests=Dict{String,String}()
    # Complete the validation pass before writing any migrated checkpoints.
    for (k,path) in enumerate(files)
        validate_transfer_checkpoint(path,old_hashes,inputs)
        digests[path]=bytes2hex(sha256(read(path)))
        k%25==0 && (println("Validated $k / $(length(files)) completed pairs");flush(stdout))
    end
    mkpath(target)
    for path in files
        digest=digests[path]
        @assert bytes2hex(sha256(read(path)))==digest "Source changed during recovery"
        dest=joinpath(target,basename(path))
        if isfile(dest)
            existing=validate_transfer_checkpoint(dest,hashes,inputs)
            @assert existing["recovery_provenance"].source_sha256==digest
            continue
        end
        d=load(path);d["configuration"]=merge(d["configuration"],(;code_hashes=hashes))
        d["recovery_provenance"]=(;source_signature=signature,source_sha256=digest,
            source_code_hashes=old_hashes,reason="Only redundant Pkg activation removed; numerical code and inputs unchanged")
        temporary=dest*".migration.tmp"
        jldsave(temporary; (Symbol(k)=>v for (k,v) in d)...)
        Base.Filesystem.rename(temporary,dest)
    end
    total=count(p->startswith(p,"res_ADD_entropic_") && endswith(p,".jld2"),readdir(target))
    println("Recovery complete: $(length(files)) original pairs preserved; $total completed pairs in destination.")
    println("At most $(945-total) pairs remain. Destination: $target")
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    length(ARGS)==2 || error("Usage: julia --project=. scripts/resume_ADD_locality_transfer.jl SOURCE_HASH EXPECTED_COMPLETED")
    resume_transfer(ARGS[1],parse(Int,ARGS[2]))
end
