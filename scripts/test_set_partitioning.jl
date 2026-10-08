#!/usr/bin/env julia
# Generator tests require only Julia's standard library, not the ML stack.
using Random, Test
include(joinpath(@__DIR__,"..","src","set_partitioning.jl"))

@testset "Exact covers, independent exhaustive oracle" begin
    tiles = [[1],[2],[3],[4],[1,2],[3,4],[1,3],[2,4],[1,2,3,4]]
    A = [Int(u in t) for u in 1:4,t in tiles]
    valid = Set{Tuple}()
    for mask in 0:2^length(tiles)-1
        x = [Int((mask >> (j-1)) & 1) for j in eachindex(tiles)]
        all(A*x .== 1) && push!(valid,Tuple(x))
    end
    sampler = ExactPartitionSampler(tiles,4)
    @test sampler.counts[UInt128(0)] == length(valid) == 8
    X = ones(Int,length(tiles),32000)
    fill_partition_vectors!(MersenneTwister(781),X,sampler)
    @test Set(Tuple(x) for x in eachcol(X)) == valid
    frequencies = Dict(x=>0 for x in valid)
    foreach(x->frequencies[Tuple(x)]+=1,eachcol(X))
    @test all(abs(c/size(X,2)-1/8) < 0.015 for c in values(frequencies))
    Y = zeros(Int,length(tiles),50)
    fill_partition_vectors!(MersenneTwister(781),Y,sampler)
    @test Y == X[:,1:50]
    buffer = ones(Int,length(tiles),52)
    fill_partition_vectors!(MersenneTwister(781),@view(buffer[:,2:51]),sampler)
    @test buffer[:,2:51] == Y
    @test all(buffer[:,[1,52]] .== 1)
    permutation = [9,3,2,8,4,6,1,7,5]
    permuted = ExactPartitionSampler(tiles[permutation],4)
    @test permuted.counts[UInt128(0)] == length(valid)
    Z = zeros(Int,9,100)
    fill_partition_vectors!(MersenneTwister(9),Z,permuted)
    @test all(A[:,permutation]*Z .== 1)
    @test_throws ArgumentError ExactPartitionSampler([[1,2],[2,3]],3)
end

@testset "Production geometry and explicit RNG" begin
    for n in 12:16, density in (:light,:balanced,:heavy)
        p = strip_partition_instance(MersenneTwister(782),n;density)
        q = strip_partition_instance(MersenneTwister(782),n;density)
        @test p == q
        @test size(p.A) == (3n,n^2)
        @test allunique(p.tiles)
        @test all(s->1<=s<=4,p.column_sizes)
        s = ExactPartitionSampler(p.tiles,3n)
        X, Y = zeros(Int,n^2,50),zeros(Int,n^2,50)
        fill_partition_vectors!(MersenneTwister(3),X,s)
        Random.seed!(123); rand(1000) # unrelated model RNG must not affect source
        fill_partition_vectors!(MersenneTwister(3),Y,s)
        @test X == Y
        @test all(p.A*X .== p.b)
        @test s.counts[UInt128(0)] > 0
    end
    rng = MersenneTwister(4)
    limit = big(2)^150+57
    draws = [partition_randbelow(rng,limit) for _ in 1:100]
    @test all(x->0<=x<limit,draws)
    @test length(unique(draws)) == 100
end
