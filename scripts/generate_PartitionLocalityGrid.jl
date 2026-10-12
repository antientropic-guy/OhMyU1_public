#!/usr/bin/env julia
include("generate_PartitionLocality.jl")
const LOCALITY_GRID_ROOT=joinpath(PARTITION_ROOT,"data","set_partitioning","locality_grid_v1")
grid_path(w,m,n,i)=joinpath(LOCALITY_GRID_ROOT,"instances","w$(w)_m$(m)_n$(n)_i$(i).jld2")
function generate_grid()
    rows=NamedTuple[]
    for w in LOCALITY_SPANS,m in 18:20,n in 13:15
        for i in 1:15
            seed=1_900_000_000+1_000_000w+10_000m+100n+i
            path=grid_path(w,m,n,i)
            specification=(;version=1,span=w,elements=m,n,variables=n^2,i,seed,
                initial_draws=500,scenarios=32,tau=.5,
                generator_sha=bytes2hex(sha256(read(joinpath(PARTITION_ROOT,"src","partition_locality.jl")))),
                column_order="minimum element; singleton last",julia_version=string(VERSION))
            if isfile(path)
                d=load(path);@assert d["specification"]==specification
                instance=d["instance"];structural=d["structural"];initial=d["initial"]
            else
                instance=locality_instance(MersenneTwister(seed),w;elements=m,n)
                structural=locality_ranks(instance)
                sampler=ExactPartitionSampler(instance.tiles,m)
                @assert structural.total==sampler.counts[UInt128(0)]
                initial=zeros(Int,n^2,500)
                fill_partition_vectors!(MersenneTwister(seed+100_000_000),initial,sampler)
                rng=MersenneTwister(seed+200_000_000)
                objectives=(;c=2rand(rng,n^2).-1,scenarios=randn(rng,32,n^2),tau=.5)
                mkpath(dirname(path));initial=UInt8.(initial)
                jldsave(path*".tmp";specification,instance,structural,initial,objectives,dp_states=length(sampler.counts))
                Base.Filesystem.rename(path*".tmp",path)
            end
            # Validate every instance, including resumed files, independently.
            @assert size(instance.A)==(m,n^2) && allunique(Tuple.(instance.tiles))
            @assert all(instance.A*Int.(initial) .== instance.b)
            @assert all(x->x in (0,1),initial)
            sampler=ExactPartitionSampler(instance.tiles,m)
            @assert structural.total==sampler.counts[UInt128(0)]
            replay=zeros(Int,n^2,500)
            fill_partition_vectors!(MersenneTwister(seed+100_000_000),replay,sampler)
            @assert replay==initial
            # Independent fresh stream, not merely a replay of saved examples.
            fill_partition_vectors!(MersenneTwister(seed+400_000_000),replay,sampler)
            @assert all(instance.A*replay .== instance.b)
            push!(rows,(;w,m,n,i,maxrank=structural.maxrank,meanrank=structural.meanrank,
                covers=string(structural.total),sha256=bytes2hex(sha256(read(path)))))
            GC.gc()
        end
        println("Validated w=$w m=$m n=$n (15 instances)");flush(stdout)
    end
    jldsave(joinpath(LOCALITY_GRID_ROOT,"manifest.jld2");rows)
    open(joinpath(LOCALITY_GRID_ROOT,"rank_instances.csv"),"w") do io
        println(io,join(keys(first(rows)),","));foreach(r->println(io,join(values(r),",")),rows)
    end
    println("945 instances validated: exact counts, feasibility, fresh samples and RNG replay.")
end
if abspath(PROGRAM_FILE)==@__FILE__; generate_grid(); end

