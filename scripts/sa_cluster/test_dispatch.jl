# Test worker dispatch/checkpoint/resume with mocked optimizers, never a benchmark.
include("run.jl")
using Test

const mock_calls=Ref(0)
function mock_trial(input;objective)
    mock_calls[]+=1
    costs=[objective(x) for x in eachcol(input.initial)]
    i=argmin(costs);v=costs[i]
    Dict{String,Any}("initial_c_min"=>v,"c_min"=>v,"solution"=>copy(input.initial[:,i]),
        "history"=>[(0.,v)],"evaluation_history"=>[(length(costs),v)],
        "objective_calls"=>length(costs),"elapsed_seconds"=>0.,
        "completed_iterations"=>0,"iterations"=>0,"generated"=>0,"initialization_seconds"=>0.)
end
# Deliberately replace the imported adapters only in this isolated test process.
sa_trial(input,budget;objective,kwargs...)=mock_trial(input;objective)
u1_trial(input,budget;objective,kwargs...)=mock_trial(input;objective)

@testset "Mocked worker resume and protocol checks" begin
    mktempdir() do output
        run_worker(;job=0,output,budgets=[1],count=1,worker=1,workers=2)
        path=joinpath(output,"assignment_quadratic_worker1.jld2")
        d=load(path)
        @test length(d["results"])==6
        @test !isdir(path*".lock")
        calls=mock_calls[]
        run_worker(;job=0,output,budgets=[1],count=1,worker=1,workers=2)
        @test mock_calls[]==calls
        @test_throws ErrorException run_worker(;job=0,output,budgets=[2],count=1,worker=1,workers=2)
        mkdir(path*".lock")
        @test_throws ErrorException run_worker(;job=0,output,budgets=[1],count=1,worker=1,workers=2)
        rm(path*".lock")
        run_worker(;job=0,output,budgets=[1],count=1,worker=2,workers=2)
        d2=load(joinpath(output,"assignment_quadratic_worker2.jld2"))
        @test isempty(intersect(Set(keys(d["results"])),Set(keys(d2["results"]))))
        @test length(d2["results"])==3
        # Check the actual 32-way partition, including an empty shard at count=1.
        entries=[(n,i) for i in 1:40 for n in (10,12,14)]
        pieces=[entries[w:32:end] for w in 1:32]
        @test length(unique(vcat(pieces...)))==120
        @test sort(length.(pieces))==vcat(fill(3,8),fill(4,24))
        run_worker(;job=0,output=joinpath(output,"empty"),budgets=[1],count=1,worker=32,workers=32)
        @test isempty(load(joinpath(output,"empty","assignment_quadratic_worker32.jld2"),"results"))
    end
end
