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
        run_worker(;job=0,output,budgets=[1],count=1)
        path=joinpath(output,"assignment_n12.jld2")
        d=load(path)
        @test length(d["results"])==6
        @test !isdir(path*".lock")
        calls=mock_calls[]
        run_worker(;job=0,output,budgets=[1],count=1)
        @test mock_calls[]==calls
        @test_throws ErrorException run_worker(;job=0,output,budgets=[2],count=1)
        mkdir(path*".lock")
        @test_throws ErrorException run_worker(;job=0,output,budgets=[1],count=1)
        rm(path*".lock")
    end
end
