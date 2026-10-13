# Small functional checks only; not a performance experiment.
include("run.jl")
using Test

@testset "Exact structures and shared objectives" begin
    n=3;rng=MersenneTwister(71)
    assignment=assignment_matrix(n)
    tiles=vcat([[i] for i in 1:6],[[1,2],[3,4],[5,6]])
    sort!(tiles;by=t->(first(t),length(t)==1 ? 1 : 0,Tuple(t)))
    partition=[Int(i in t) for i in 1:6,t in tiles]
    p=(A=partition,b=ones(Int,6),tiles=tiles,n=n)
    ranks,total=partition_profile(p)
    @test total==8
    @test maximum(assignment_profile(14))==6434
    @test maximum(assignment_profile(12))==1715
    for family in ("assignment","partition")
        initial=zeros(Int,9,400)
        if family=="assignment"
            for x in eachcol(initial), (i,j) in enumerate(randperm(rng,n));x[(i-1)*n+j]=1;end
            A=assignment;b=ones(Int,6);support=big(6)
        else
            A=partition;b=ones(Int,6);support=total
            fill_partition_vectors!(rng,initial,ExactPartitionSampler(tiles,6))
        end
        record=(family=family,n=n,repetition=1,seed=91,A=A,b=b,initial=initial,Q=randn(rng,9,9),
            entropic=(c=randn(rng,9),scenarios=randn(rng,32,9),tau=0.5,scale=sqrt(6.)),
            tiles=tiles,support=string(support))
        ctx=adapter_context(record)
        model=family=="assignment" ? full_indicator(n) : ctx.factory()
        family=="assignment" && canonical_full!(model,n)
        if family=="assignment"
            @test [q.Card for q in model.LinkIndices]==assignment_profile(n)
        else
            @test [q.Card for q in model.LinkIndices]==ranks
        end
        nonzero=0
        for mask in 0:511
            x=[Int((mask>>(j-1))&1) for j in 1:9]
            valid=all(A*x .== b)
            amplitude=model[x]
            @test isapprox(amplitude,valid ? inv(sqrt(Float64(support))) : 0.;atol=1e-10)
            nonzero+=valid
        end
        @test nonzero==support
        for kind in ("quadratic","entropic")
            objective=make_objective(record,kind)
            for x in eachcol(initial[:,1:10])
                expected=if kind=="quadratic"
                    dot(x,record.Q*x)
                else
                    o=record.entropic
                    dot(o.c,x)/o.scale+o.tau*log(mean(exp.((o.scenarios*x)./(o.tau*o.scale))))
                end
                @test isapprox(objective(x),expected;atol=1e-10)
            end
            r=sa_trial(ctx.input,Inf;objective,A,neighbor=ctx.neighbor,restarts=2,max_steps_per_chain=50)
            @test all(A*r["solution"] .== b)
            @test r["c_min"]<=r["initial_c_min"]
            for full in (false,true)
                r=u1_trial(ctx.input,Inf;objective,A,full_factory=ctx.factory,full,maxiter=1,samples_per_iteration=100)
                @test all(A*r["solution"] .== b)
                @test r["completed_iterations"]==1
                @test r["objective_calls"]>=100
                @test isapprox(objective(r["solution"]),r["c_min"];atol=1e-10)
            end
        end
    end
end

@testset "Frozen snapshot" begin
    folder=joinpath(ROOT,"data","sa_cluster_v1")
    if isfile(joinpath(folder,"manifest.jld2"))
        manifest=load(joinpath(folder,"manifest.jld2"),"manifest")
        @test length(manifest.records)==160
        for entry in manifest.records
            path=joinpath(folder,"inputs",entry.file)
            @test bytes2hex(sha256(read(path)))==entry.sha256
            record=load(path,"record")
            @test all(record.A*record.initial .== record.b)
            if entry.family=="partition"
                paired=load(joinpath(folder,"inputs","assignment_n$(entry.n)_i$(entry.instance).jld2"),"record")
                @test paired.Q==record.Q
                @test paired.entropic==record.entropic
                @test entry.maxrank<100 # Audit the intended low-rank regime; do not filter tasks.
            end
        end
    end
end
