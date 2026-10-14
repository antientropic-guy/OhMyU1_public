# Small local correctness tests; no full-sized benchmark or cluster submission.
include("run_penalty.jl")
using Test

@testset "Generic binary penalty baseline" begin
    A=sparse(reshape([1,1,1,1],1,4));b=[2]
    points=[Int[(mask>>(j-1))&1 for j in 1:4] for mask in 0:15]
    initial=hcat(filter(x->A*x==b,points)...)
    Q=[-3. 1. 0. 0.;1. -2. 0. 0.;0. 0. 1. 0.;0. 0. 0. 2.]
    record=(A=A,b=b,initial=initial,Q=Q,seed=19)
    objective=BinaryQuadratic(Q,Int[])
    costs=[objective(x) for x in eachcol(initial)]
    p=penalty_parameters(costs,A)
    @test p.alpha==10*(maximum(costs)-minimum(costs))
    @test p.initial_temperature>p.final_temperature>0
    @test exp(-p.alpha*p.typical_flip_violation/p.initial_temperature)≈.5
    # On this exhaustively enumerable problem the chosen penalty dominates.
    @test all(objective(x)+p.alpha*sum(abs2,A*x-b)>minimum(costs) for x in points if A*x!=b)
    rng=MersenneTwister(41)
    for x in points, _ in 1:10
        y=similar(x);bitflip_neighbor!(x,y,rng)
        @test sum(x.!=y)==1 && all(v->v in (0,1),y)
    end
    warm=penalty_trial(record,Inf;objective,restarts=1,max_steps_per_chain=100)
    r=penalty_trial(record,.15;objective)
    @test r["objective_calls"]==r["feasible_calls"]+r["infeasible_calls"]
    @test r["infeasible_calls"]>0 && r["generated"]>0
    @test A*r["solution"]==b && r["c_min"]<=minimum(costs)
    @test all(first(h)<=.15 for h in r["history"])
    @test issorted(first.(r["history"])) && all(diff(last.(r["history"])).<=0)
    @test last(r["history"])[2]==r["c_min"]
    @test last(r["call_checkpoints"])[2]==r["objective_calls"]
    @test r["penalty_queries"]==r["objective_calls"]-length(costs)
    @test penalty_parameters(ones(4),A).alpha>0
    @test isfinite(penalty_parameters(zeros(4),spzeros(Int,1,4)).initial_temperature)
end

# Replace only this process's trial with a deterministic mock. Loading and
# dispatch use actual frozen inputs, but no optimization is performed.
function penalty_trial(record,budget;objective,kwargs...)
        c=[objective(x) for x in eachcol(record.initial)];j=argmin(c)
        Dict{String,Any}("c_min"=>c[j],"initial_c_min"=>c[j],"solution"=>record.initial[:,j],
            "history"=>[(0.,c[j])],"evaluation_history"=>[(length(c),c[j])],"alpha"=>1.,
            "objective_calls"=>length(c),"feasible_calls"=>length(c))
end
@testset "Private checkpoint dispatch and resume" begin
    mktempdir() do out
        run_penalty_worker(;job=0,output=out,budgets=[1],count=1,workers=2,worker=1)
        path=joinpath(out,"assignment_quadratic_worker1.jld2")
        d=load(path);@test length(d["results"])==2 && !isdir(path*".lock")
        previous=read(path)
        run_penalty_worker(;job=0,output=out,budgets=[1],count=1,workers=2,worker=1)
        @test read(path)==previous
        @test_throws ErrorException run_penalty_worker(;job=0,output=out,budgets=[2],count=1,workers=2,worker=1)
        mkdir(path*".lock")
        @test_throws ErrorException run_penalty_worker(;job=0,output=out,budgets=[1],count=1,workers=2,worker=1)
        rm(path*".lock")
        run_penalty_worker(;job=0,output=out,budgets=[1],count=1,workers=2,worker=2)
        d2=load(joinpath(out,"assignment_quadratic_worker2.jld2"))
        @test length(d2["results"])==1
        @test isempty(intersect(Set(keys(d["results"])),Set(keys(d2["results"]))))
    end
end

@testset "Collector completeness" begin
    mktempdir() do out
        for f in ("assignment","partition"),o in ("quadratic","entropic")
            protocol=(version=1,family=f,objectives=[o],worker=1,workers=1,count=1,sizes=[10,12,14],
                methods=["SA_penalty"],budgets=[1],assigned=[(n,1) for n in (10,12,14)],
                manifest_sha256="toy",sources=Dict("toy"=>"hash"),alpha_rule="toy",temperature="toy",proposal="toy",
                julia_version=string(VERSION),optim_version=string(pkgversion(Optim)))
            r=Dict{String,Any}(k=>0. for k in ["initial_c_min","c_min","elapsed_seconds","objective_calls","completed_iterations","alpha","initial_temperature","final_temperature","feasible_calls","infeasible_calls","best_penalized","best_penalized_violation","infeasible_beats_initial","max_violation"])
            r["history"]=[(0.,0.)];r["evaluation_history"]=[(0,0.)];r["call_checkpoints"]=[(0.,0,0,0)]
            results=Dict((n,1,o,1,"SA_penalty")=>r for n in (10,12,14))
            atomic_save(joinpath(out,"$(f)_$(o)_worker1.jld2");results,protocol)
        end
        saved_args=copy(ARGS);empty!(ARGS);push!(ARGS,out)
        try
            Base.include(Module(:PenaltyCollectorCheck),joinpath(@__DIR__,"collect_penalty.jl"))
            @test length(load(joinpath(out,"combined.jld2"),"results"))==12
            @test length(readlines(joinpath(out,"summary.csv")))==13
            p=joinpath(out,"assignment_quadratic_worker1.jld2");d=load(p)
            delete!(d["results"],first(keys(d["results"])))
            atomic_save(p;results=d["results"],protocol=d["protocol"])
            @test_throws LoadError Base.include(Module(:PenaltyCollectorIncomplete),joinpath(@__DIR__,"collect_penalty.jl"))
        finally
            empty!(ARGS);append!(ARGS,saved_args)
        end
    end
end
