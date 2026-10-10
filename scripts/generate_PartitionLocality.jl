#!/usr/bin/env julia
isdefined(@__MODULE__,:PARTITION_ROOT) || include("generate_set_partitioning_instances.jl")
include(joinpath(PARTITION_ROOT,"src","partition_locality.jl"))
using OhMyU1: fill_partition_vectors!
using Statistics
const LOCALITY_SPANS=[5,6,8,10,12,15,18]
const LOCALITY_ROOT=joinpath(PARTITION_ROOT,"data","set_partitioning","locality_v1")
locality_path(span,i)=joinpath(LOCALITY_ROOT,"instances","span$(span)_i$(i).jld2")
locality_seed(span,i)=1_710_000_000+10_000span+i
function generate_locality(;instances=1:40,spans=LOCALITY_SPANS)
    for span in spans,i in instances
        path=locality_path(span,i)
        generator_sha=bytes2hex(sha256(read(joinpath(PARTITION_ROOT,"src","partition_locality.jl"))))
        specification=(;version=1,n=15,elements=18,variables=225,span,i,
            seed=locality_seed(span,i),generator_sha,initial_draws=500,scenarios=32,tau=.5,
            column_order="minimum element; singleton last",julia_version=string(VERSION))
        if isfile(path)
            load(path,"specification")==specification || error("Generator mismatch: $path")
            continue
        end
        instance=locality_instance(MersenneTwister(specification.seed),span)
        structural=locality_ranks(instance)
        sampler=ExactPartitionSampler(instance.tiles,length(instance.b))
        @assert structural.total==sampler.counts[UInt128(0)]
        initial=zeros(Int,225,500)
        fill_partition_vectors!(MersenneTwister(specification.seed+100_000_000),initial,sampler)
        @assert all(instance.A*initial .== instance.b)
        rng=MersenneTwister(specification.seed+200_000_000)
        objectives=(;c=2rand(rng,225).-1,scenarios=randn(rng,32,225),tau=.5)
        # Prefix-charge coverage by the raw initial sample, before cost selection.
        q=zeros(UInt32,500);observed=Int[]
        for t in instance.tiles
            j=length(observed)+1
            mask=foldl(|,(UInt32(1)<<(u-1) for u in t);init=UInt32(0))
            for k in 1:500; initial[j,k]==1 && (q[k]|=mask); end
            push!(observed,length(unique(q)))
        end
        initial_coverage=sum(observed[1:end-1])/sum(structural.ranks[2:end-1])
        mkpath(dirname(path))
        initial=UInt8.(initial)
        jldsave(path*".tmp";specification,instance,objectives,initial,structural,
            observed_ranks=observed,initial_coverage,dp_states=length(sampler.counts))
        Base.Filesystem.rename(path*".tmp",path)
        println("span=$span i=$i max=$(structural.maxrank) mean=$(round(structural.meanrank;digits=1)) covers=$(structural.total)")
        flush(stdout);GC.gc()
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    all(x->x=="--pilot",ARGS) || error("Only --pilot is supported")
    generate_locality(;instances="--pilot" in ARGS ? (1:1) : (1:40))
end
