using Test
include("run.jl")
@testset "Deadline accounting" begin
    r=Recorder(time_ns(),Inf,0,Inf,Int[],NamedTuple[],NamedTuple[],0,NamedTuple[],
        [1.,2.],0,Dict{Float64,Int}())
    r.completed_calls=7
    capture_calls(r,1.5)
    r.completed_calls=9
    capture_calls(r,2.5)
    @test r.budget_calls==Dict(1.0=>7,2.0=>9)
    r.limit=0.
    @test_throws BudgetExpired counted(r,x->0.)([1])
    @test r.calls==0
    @test isempty(r.improvements)
end
@testset "Dispatch and paired production equivalence" begin
    tt=tasks()
    shards=[tt[j*32+k+1:128:end] for j in 0:3 for k in 0:31]
    @test length(vcat(shards...))==1945
    @test allunique(getproperty.(vcat(shards...),:id))
    for mode in ("add","eels","transfer"), objective in (mode=="transfer" ? ("entropic",) : ("linear","quadratic"))
        t=first(filter(t->t.mode==mode && t.objective==objective,tt))
        d=inputs(t;small=true)
        @test all(d.A*d.initial .== d.b)
        initial_hash=E.training_fingerprint(d.initial)
        for arm in ("best_cost",mode=="eels" ? "eels" : "gdf")
            timed=execute(t,d,arm,Inf;small=true,maxiter=2)
            sp=timed.parameters
            problem=U.OptimizationProblem(A=d.A,b=d.b,cost_function=d.raw)
            Random.seed!(d.seed+10_000_000)
            reference=if mode=="add"
                U.solve(problem,assignment_source(d.sampler,d.seed,d.initial),sp,
                    arm=="gdf" ? "best nonzero mixed" : "best_cost";
                    parallel=false,print_stats=false,diversify=false,barrier=false,
                    mixed_proportion=.4,rank_weight=.4)
            elseif mode=="eels"
                first(E.run_persistent_arm(problem,d.initial,sp,d.seed;eels=arm=="eels"))
            else
                Protocols.locality_add_arm(problem,d.sampler,d.initial,sp,d.seed,
                    arm=="gdf" ? "best nonzero mixed" : "best_cost";external=48).stats
            end
            @test timed.recorder.best≈reference.c_min atol=1e-12
            @test timed.initial_cost≈minimum(d.raw(x) for x in eachcol(d.initial)) atol=1e-12
            @test E.training_fingerprint(d.initial)==initial_hash
            @test length(timed.recorder.iterations)==2
            @test issorted([-x.cost for x in timed.recorder.improvements])
            @test timed.termination==:iteration_limit
        end
    end
end
