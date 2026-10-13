# Check dictionary round-trips and collection on synthetic records only.
using Test, JLD2, SparseArrays

@testset "Synthetic shard collection" begin
    mktempdir() do folder
        methods=["SA","U1_data_EELS","U1_full"]
        objectives=["quadratic","entropic"]
        for family in ("assignment","partition"), o in objectives, worker in 1:2
            assigned=[(n,1) for n in (10,12,14)][worker:2:end]
            protocol=(version=2,family=family,sizes=[10,12,14],worker=worker,workers=2,assigned=assigned,
                count=1,budgets=[10],methods=methods,objectives=[o],
                manifest_sha256="synthetic",sources=Dict("test"=>"synthetic"),julia_version=string(VERSION),optim_version="synthetic")
            results=Dict((n,i,o,10,m)=>Dict("initial_c_min"=>0.,"c_min"=>-1.,"elapsed_seconds"=>10.,
                "objective_calls"=>10,"completed_iterations"=>1,"initialization_seconds"=>0.1,
                "allocated_bytes"=>0,"gc_seconds"=>0.,"process_maxrss_bytes"=>0,
                "history"=>[(0.,0.),(1.,-1.)],"evaluation_history"=>[(1,0.),(10,-1.)]) for (n,i) in assigned for m in methods)
            path=joinpath(folder,"$(family)_$(o)_worker$(worker).jld2")
            jldsave(path,true,IOStream;results,protocol)
            @test load(path,"results")==results
        end
        original=copy(ARGS)
        try
            empty!(ARGS);push!(ARGS,folder)
            Base.include(Module(:SyntheticCollection),joinpath(@__DIR__,"collect.jl"))
            @test length(load(joinpath(folder,"combined.jld2"),"results"))==36
            @test length(readlines(joinpath(folder,"summary.csv")))==37
            @test length(readlines(joinpath(folder,"curves.csv")))==73
            # An interrupted worker must not silently enter an aggregate.
            path=joinpath(folder,"assignment_quadratic_worker1.jld2")
            d=load(path);delete!(d["results"],first(keys(d["results"])))
            jldsave(path,true,IOStream;results=d["results"],protocol=d["protocol"])
            @test_throws LoadError Base.include(Module(:IncompleteCollection),joinpath(@__DIR__,"collect.jl"))
        finally
            empty!(ARGS);append!(ARGS,original)
        end
    end
end
