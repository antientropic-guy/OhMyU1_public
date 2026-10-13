# Extend the frozen snapshot without modifying its n=12,14 inputs.
include("prepare.jl")
using OhMyU1: strip_partition_instance

function extend_n10(;out=joinpath(ROOT,"data","sa_cluster_v2"))
    ispath(out) && error("Refusing to overwrite $out")
    sources=joinpath(out,"sources_n10")
    n=10
    for rep in 1:40
        # Same SP generator, objective streams and seed convention as the old test set.
        seed=811_000_000+1000n+rep
        instance=strip_partition_instance(MersenneTwister(seed),n;density=:light)
        rng=MersenneTwister(seed+200_000_000)
        objectives=(c=2rand(rng,n*n).-1,scenarios=randn(rng,32,n*n),tau=0.5)
        specification=(seed=seed,n=n,i=rep,density="light",split="test")
        sampler=ExactPartitionSampler(instance.tiles,length(instance.b))
        exact_cover_count=sampler.counts[UInt128(0)]
        spath=joinpath(sources,"data","set_partitioning","test","instances","light","instance_$(n)_$(rep)_test.jld2")
        mkpath(dirname(spath))
        atomic_save(spath;instance,objectives,specification,exact_cover_count)
        # Independent fixed per-instance stream; the same symmetrized U[-7,7]
        # matrix construction as generate_assignment_test_instances.jl.
        qseed=912_010_000+rep
        raw=14rand(MersenneTwister(qseed),n*n,n*n).-7
        q=sparse((raw+raw')/2)
        metadata=(seed=qseed,n=n,instance=rep,split="test",radius=7)
        qpath=joinpath(sources,"data","test_random_bilinear_forms","q_$(n)_$(rep)_r=7_test.jld2")
        mkpath(dirname(qpath));atomic_save(qpath;q,metadata)
    end
    prepare(;out,sizes=(10,),count=40,source_root=sources,
        previous=joinpath(ROOT,"data","sa_cluster_v1"))
end

abspath(PROGRAM_FILE)==(@__FILE__) && extend_n10()
