# Load the actual core source without graphics/solver dependencies. This small
# subprocess tests only the valid no-interactive-pool runtime configuration.
using Test, Random, LinearAlgebra, StatsBase, OMEinsum
module CoreUnderTest
using Random, LinearAlgebra, StatsBase, OMEinsum
include(joinpath(@__DIR__,"..","src","mps_core.jl"))
end
const U=CoreUnderTest
@testset "Parallel sampler without interactive pool" begin
    A=ones(Int,1,4);b=[2];T=[1 0;1 0;0 1;0 1];domains=fill([0,1],4)
    lc=U.compute_link_charges(A,b,T);sc=U.compute_site_charges(A,domains)
    links,sites=U.compute_indices(lc,sc,b,domains,A)
    m=U.init_u1_mps(Float64,links,sites,A,b,1,fill(2,4));U.orthogonalize!(m);U.normalize!(m)
    X=zeros(Int,4,128);U.sample_nondeg_parallel!(m,X,128)
    @test all(A*X .== b)
    for count in (0,1,2,3,7,129)
        buf=fill(-1,4,count+2)
        U.sample_nondeg_parallel!(m,buf,count)
        @test all(A*buf[:,1:count] .== b)
        @test all(buf[:,count+1:end] .== -1)
    end
    @test_throws ArgumentError U.sample_nondeg_parallel!(m,X,-1)
    @test_throws DimensionMismatch U.sample_nondeg_parallel!(m,X,129)
    # Calls nested in independently scheduled tasks share a read-only model,
    # but must never share scratch arrays or output buffers.
    tasks=[Threads.@spawn begin
        yield()
        buf=zeros(Int,4,513)
        U.sample_nondeg_parallel!(m,buf,513)
        buf
    end for _ in 1:8]
    @test all(all(A*fetch(t) .== b) for t in tasks)
end
