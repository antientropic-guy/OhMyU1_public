# Recovery regression checks use synthetic results on one real frozen input.
# No optimization or modification of real result folders is performed.
using Test
include("resume_ADD_locality_transfer.jl")

@testset "Activation-only ADD recovery" begin
    hashes=merge(locality_code_hashes(),Dict("scripts/run_ADD_locality_transfer.jl"=>bytes2hex(sha256(read(joinpath(@__DIR__,"run_ADD_locality_transfer.jl"))))))
    old=copy(hashes);old["scripts/run_EELS_optimization.jl"]=RESUME_OLD_EELS_HASH
    signature=transfer_signature(old)
    sp=SolverParams(NUM_FEASIBLE_SAMPLES=500,NUM_MPS_SAMPLES=5000,NUM_GLOBAL_ITER=15,
        NUM_SWEEP_ITER=1,LINK_DEGENERACY=1,LEARNING_RATE=.05,UTILITY_FRACTION=.05,KEEP_NUM_WORST=0.)
    parameters=NamedTuple{fieldnames(SolverParams)}(Tuple(getfield(sp,f) for f in fieldnames(SolverParams)))
    cfg=(;protocol=3,span=5,elements=18,n=13,objective="entropic",grid=true,mode="ADD",
        external=5000,parameters,code_hashes=old,gamma=1,filter_weight=.4,
        sampling_backend="table",julia_version=string(VERSION),
        incumbent="best of all evaluated candidates including final external batch")
    input=grid_path(5,18,13,1);d=load(input);initial=Int.(d["initial"][:,1:500])
    objective=partition_cost(d["objectives"],"entropic",18)
    T,costs=sort_and_truncate(initial,objective,500)
    stats=SolverStatistics();stats.incub=copy(T[:,1]);stats.c_min=first(costs)
    stats.learning_curve=fill(stats.c_min,16);stats.utility_curve=fill(stats.c_min,16)
    arm=(;initial_training_sha256=PartitionEELS.training_fingerprint(T),
        observations=[(;iteration=k,external_draws=5000,external_sha256="synthetic-paired-source") for k in 1:15])
    diag=(;input_sha256=bytes2hex(sha256(read(input))),initial_sha256=PartitionEELS.training_fingerprint(initial),
        structural=d["structural"],seed=d["specification"].seed+100_000_000,
        arms=Dict("best_cost"=>arm,"best nonzero mixed"=>arm))
    inputs=Dict((5,18,13,1)=>diag.input_sha256)
    mktempdir() do root
        source=joinpath(root,signature,"entropic");mkpath(source)
        path=joinpath(source,"res_ADD_entropic_span5_m18_n13_i1.jld2")
        res=Dict("best_cost"=>Dict(13=>Dict(1=>stats)),"best nonzero mixed"=>Dict(.4=>Dict(13=>Dict(1=>stats))))
        jldsave(path;res_dict=res,solver_params=sp,configuration=cfg,diagnostics=Dict(1=>diag))
        original=read(path)
        @test validate_transfer_checkpoint(path,old,inputs)["configuration"]==cfg
        @test_throws AssertionError validate_transfer_checkpoint(path,hashes,inputs)
        @test_throws AssertionError validate_transfer_checkpoint(path,old,Dict((5,18,13,1)=>"wrong-input"))
        @test_throws ErrorException resume_transfer(signature,2;root)
        @test !isdir(joinpath(root,transfer_signature(hashes)))
        resume_transfer(signature,1;root)
        dest=joinpath(root,transfer_signature(hashes),"entropic",basename(path))
        saved=validate_transfer_checkpoint(dest,hashes,inputs)
        @test saved["recovery_provenance"].source_sha256==bytes2hex(sha256(original))
        @test read(path)==original
        @test saved["res_dict"]["best_cost"][13][1].learning_curve==stats.learning_curve
        migrated=read(dest)
        resume_transfer(signature,1;root)
        @test read(dest)==migrated # Idempotent; no overwrite on another recovery attempt.
        changed=copy(old);changed["src/train.jl"]="unexpected-code"
        @test_throws AssertionError validate_transfer_checkpoint(path,changed,inputs)
        jldsave(path;res_dict=res,solver_params=sp,configuration=merge(cfg,(;external=1000)),diagnostics=Dict(1=>diag))
        @test_throws AssertionError validate_transfer_checkpoint(path,old,inputs)
        stats.learning_curve=fill(stats.c_min,15)
        jldsave(path;res_dict=res,solver_params=sp,configuration=cfg,diagnostics=Dict(1=>diag))
        @test_throws AssertionError validate_transfer_checkpoint(path,old,inputs)
    end
end
