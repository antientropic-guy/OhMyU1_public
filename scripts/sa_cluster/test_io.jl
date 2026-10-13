# Check dictionary round-trips and collection on synthetic records only.
using Test, JLD2, SparseArrays

@testset "Synthetic shard collection" begin
    mktempdir() do folder
        methods=["SA","U1_data_EELS","U1_full"]
        objectives=["quadratic","entropic"]
        for (family,n) in (("assignment",12),("assignment",14),("partition",12),("partition",14))
            protocol=(family=family,n=n,count=1,budgets=[10],methods=methods,objectives=objectives,
                manifest_sha256="synthetic",sources=Dict("test"=>"synthetic"),julia_version=string(VERSION),optim_version="synthetic")
            results=Dict((1,o,10,m)=>Dict("initial_c_min"=>0.,"c_min"=>-1.,"elapsed_seconds"=>10.,
                "objective_calls"=>10,"completed_iterations"=>1,"initialization_seconds"=>0.1,
                "allocated_bytes"=>0,"gc_seconds"=>0.,"process_maxrss_bytes"=>0,
                "history"=>[(0.,0.),(1.,-1.)],"evaluation_history"=>[(1,0.),(10,-1.)]) for o in objectives for m in methods)
            path=joinpath(folder,"$(family)_n$(n).jld2")
            jldsave(path,true,IOStream;results,protocol)
            @test load(path,"results")==results
        end
        original=copy(ARGS)
        try
            empty!(ARGS);push!(ARGS,folder)
            Base.include(Module(:SyntheticCollection),joinpath(@__DIR__,"collect.jl"))
            @test length(load(joinpath(folder,"combined.jld2"),"results"))==24
            @test length(readlines(joinpath(folder,"summary.csv")))==25
            @test length(readlines(joinpath(folder,"curves.csv")))==49
            # An interrupted worker must not silently enter an aggregate.
            path=joinpath(folder,"assignment_n12.jld2")
            d=load(path);delete!(d["results"],first(keys(d["results"])))
            jldsave(path,true,IOStream;results=d["results"],protocol=d["protocol"])
            @test_throws LoadError Base.include(Module(:IncompleteCollection),joinpath(@__DIR__,"collect.jl"))
        finally
            empty!(ARGS);append!(ARGS,original)
        end
    end
end
