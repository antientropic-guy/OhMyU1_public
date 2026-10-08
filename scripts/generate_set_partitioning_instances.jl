#!/usr/bin/env julia
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__,"..")))
using OhMyU1, Random, JLD2, SHA
using OhMyU1: strip_partition_instance, ExactPartitionSampler

const PARTITION_ROOT = normpath(joinpath(@__DIR__,".."))
partition_seed(n,i,d) = 810_000_000 + 1_000_000*findfirst(==(d),["light","balanced","heavy"]) + 1000n+i
partition_path(root,d,n,i) = joinpath(root,"instances",d,"instance_$(n)_$(i)_test.jld2")

function partition_source_hashes()
    paths = [joinpath(PARTITION_ROOT,"src",f) for f in sort(readdir(joinpath(PARTITION_ROOT,"src"))) if endswith(f,".jl") && !(f in ("deprecated.jl","future.jl","py_tools.jl"))]
    append!(paths,[joinpath(@__DIR__,f) for f in ("generate_set_partitioning_instances.jl","run_SetPartitioning.jl","run_EELS_optimization.jl")])
    Dict(relpath(p,PARTITION_ROOT)=>bytes2hex(sha256(read(p))) for p in paths)
end

function generate_partition_instances(root; ns=12:16, instances=1:50,
        densities=["light"])
    generator_hash = bytes2hex(sha256(read(joinpath(PARTITION_ROOT,"src","set_partitioning.jl"))))
    script_hash = bytes2hex(sha256(read(@__FILE__)))
    for d in densities, n in ns, i in instances
        path = partition_path(root,d,n,i)
        seed = partition_seed(n,i,d)
        specification = (; version=2,split="test",seed,n,i,density=d,width=3,variables=n^2,
            generator_hash,script_hash,scenarios=32,tau=0.5,julia_version=string(VERSION))
        if isfile(path)
            load(path,"specification") == specification || error("Instance specification mismatch: $path")
            continue
        end
        instance = strip_partition_instance(MersenneTwister(seed),n;density=Symbol(d))
        # Separate streams: objective draws cannot change the feasible space.
        rng = MersenneTwister(seed+200_000_000)
        objectives = (; c=2rand(rng,n^2).-1,
            scenarios=randn(rng,32,n^2),tau=0.5)
        sampler = ExactPartitionSampler(instance.tiles,length(instance.b))
        exact_cover_count = sampler.counts[UInt128(0)]
        dp_states = length(sampler.counts)
        mkpath(dirname(path))
        jldsave(path*".tmp";instance,objectives,specification,exact_cover_count,dp_states)
        Base.Filesystem.rename(path*".tmp",path)
    end
    @info "Test instances ready" root ns instances densities
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoke = "--smoke" in ARGS
    all(a -> a == "--smoke",ARGS) || error("Usage: generate_set_partitioning_instances.jl [--smoke]")
    root = joinpath(PARTITION_ROOT,"data","set_partitioning",smoke ? "smoke" : "test")
    generate_partition_instances(root;ns=smoke ? [12] : 12:16,
        instances=smoke ? [1] : 1:50)
end
