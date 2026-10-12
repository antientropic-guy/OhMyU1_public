@testset "Article ADD strategies: independently selected columns" begin
    f=first(fixtures());m=build(f);X=space(4);X=X[:,[f.A*x==f.b for x in eachcol(X)]]
    history=[Set(collect(q) for q in Q) for Q in observed_charges(f.A,f.b,f.T)[2:end]]
    dist=[sum(!(Tuple(f.b-f.A[:,1:k]*x[1:k]) in Set(Tuple.(history[k]))) for k in 1:4) for x in eachcol(X)]
    costs=[4.,1.,6.,2.,5.,3.];cost(x)=costs[findfirst(==(Tuple(x)),Tuple.(eachcol(X)))]
    @test [U.graph_dist_global(x,m,history) for x in eachcol(X)]==dist
    @test [U.graph_dist_global!(zeros(Int,1),x,m,history) for x in eachcol(X)]==dist
    before=deepcopy(history);raw=copy(X)
    for w in (0.,.1,.2,.3,.4,1.),strategy in ("best_cost","mixed","weighted rank","best nonzero mixed")
        sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=4,KEEP_NUM_WORST=0.)
        nc=sortperm(costs);ng=sortperm(dist;rev=true);g=round(Int,4w)
        expected = if strategy=="best_cost"
            nc[1:4]
        elseif strategy=="mixed"
            rest=ng[g+1:end];vcat(rest[sortperm(costs[rest])[1:4-g]],ng[1:g])
        elseif strategy=="weighted rank"
            score=invperm(nc)+w*invperm(ng);sortperm(score)[1:4]
        else
            first_cost=nc[1:4-g];eligible=[i for i in nc[5-g:end] if dist[i]>0]
            isempty(eligible) ? nc[1:4] : vcat(first_cost,eligible[1:min(g,length(eligible))])
        end
        selected=first(U.pick_diverse_data!(m,X,history,U.graph_dist_global!,strategy,cost,sp,false,1;mixed_proportion=w,rank_weight=w))
        @test selected==X[:,expected]
        @test all(f.A*selected .== f.b)
        @test X==raw && history==before
    end
    # All-known fallback and too-few-novel candidates are explicit policies:
    # the latter may return fewer than requested; never duplicate to fill it.
    fullhistory=[Set(collect(q) for q in Q) for Q in observed_charges(f.A,f.b,X)[2:end]]
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=4,KEEP_NUM_WORST=0.)
    chosen=first(U.pick_diverse_data!(m,X,fullhistory,U.graph_dist_global!,"best nonzero mixed",cost,sp,false,1;mixed_proportion=.4))
    @test chosen==X[:,sortperm(costs)[1:4]]
    # Barrier behavior: filter only if more than the requested number are novel.
    for wanted in (2,4)
        sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=wanted,KEEP_NUM_WORST=0.)
        eligible=count(>(0),dist)>wanted ? findall(>(0),dist) : collect(eachindex(dist))
        expected=eligible[sortperm(costs[eligible])[1:wanted]]
        got=first(U.pick_diverse_data!(m,X,history,U.graph_dist_global!,"best_cost",cost,sp,true,1))
        @test got==X[:,expected]
    end
end
@testset "Selection diagnostics and tie handling contracts" begin
    f=first(fixtures());m=build(f);X=space(4);X=X[:,[f.A*x==f.b for x in eachcol(X)]]
    history=[Set(collect(q) for q in Q) for Q in observed_charges(f.A,f.b,f.T)[2:end]]
    costs=collect(1.:6);cost(x)=costs[findfirst(==(Tuple(x)),Tuple.(eachcol(X)))]
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=4,KEEP_NUM_WORST=0.)
    got,num_new,hi,lo,_=U.pick_diverse_data!(m,X,history,U.graph_dist_global!,"best nonzero mixed",cost,sp,false,1;mixed_proportion=.4)
    ds=[U.graph_dist_global(x,m,history) for x in eachcol(X)]
    @test num_new==count(>(0),ds)
    @test (lo,hi)==extrema(ds)
    # Equal scores must not repeat the same candidate in best and worst slots.
    tied=U.pick_mixed_samples(X,3,3,ones(6))
    @test size(tied,2)==length(columns(tied))==6
end
@testset "Solver edge cases: constant objective and last external batch" begin
    function edge_case(constant)
        A=ones(Int,1,3);b=[1];calls=Ref(0)
        cost=constant ? (x->2.) : (x->Float64(-10x[1]+x[3]))
        function source!(X)
            calls[]+=1
            for k in axes(X,2)
                X[:,k].=0;X[calls[]==1 ? (iseven(k) ? 2 : 3) : 1,k]=1
            end
        end
        sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=4,NUM_MPS_SAMPLES=16,NUM_GLOBAL_ITER=1,UTILITY_FRACTION=.25,KEEP_NUM_WORST=0.)
        stats=U.solve(problem(A,b;cost),source!,sp,"best_cost";parallel=false,print_stats=false)
        @test stats.c_min==(constant ? 2. : -10.)
        @test A*stats.incub==b
    end
    for constant in (false,true)
        @testset "constant=$constant" begin
            edge_case(constant)
        end
    end
end

@testset "Diagnostics without extra graph work; ties and evaluated incumbents" begin
    f=first(fixtures());m=build(f);X=space(4);X=X[:,[f.A*x==f.b for x in eachcol(X)]]
    history=[Set(collect(q) for q in Q) for Q in observed_charges(f.A,f.b,f.T)[2:end]]
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=2,KEEP_NUM_WORST=0.)
    cost(x)=Float64(findfirst(==(Tuple(x)),Tuple.(eachcol(X))))
    no_graph(args...)=error("BestCost must not evaluate graph distance")
    _,num,hi,lo,_=U.pick_diverse_data!(m,X,history,no_graph,"best_cost",cost,sp,false,1)
    @test (num,hi,lo)==(-1,-1,-1)
    # Novel points may all be in the cost subset. Fallback diagnostics must
    # describe the evaluated pool, not just the remaining selection candidates.
    fake_dist(buf,x,args...)=cost(x)==1 ? 3 : 0
    _,num,hi,lo,_=U.pick_diverse_data!(m,X,history,fake_dist,"best nonzero mixed",cost,sp,false,1;mixed_proportion=.5)
    @test (num,hi,lo)==(1,3,0)
    for rev in (false,true), scores in (ones(6),[1.,1.,2.,2.,3.,3.]), kb in (0,2,3,6)
        got=U.pick_mixed_samples(X,kb,6-kb,scores;rev)
        @test size(got,2)==length(columns(got))==6
        @test columns(got)==columns(X)
    end
    # An evaluated cost optimum must survive even if graph ranking discards it.
    stats=U.SolverStatistics();stats.c_min=Inf
    calls=Threads.Atomic{Int}(0)
    counted(x)=(Threads.atomic_add!(calls,1);cost(x))
    reverse_dist(buf,x,args...)=Int(cost(x))
    selected=first(U.pick_diverse_data!(m,X,history,reverse_dist,"weighted rank",counted,sp,false,1;rank_weight=100.,incumbent=stats))
    @test calls[]==size(X,2)
    @test stats.c_min==1. && stats.incub==X[:,1]
    @test !(Tuple(stats.incub) in columns(selected))
end
@testset "Zero-temperature and singleton training" begin
    f=first(fixtures());sp=U.SolverParams(NUM_SWEEP_ITER=1)
    for T in (f.T,f.T[:,1:1])
        m=build(f);U.orthogonalize!(m);U.normalize!(m)
        U.train_step!(m,T,sp,0.,x->2.,2.)
        @test isfinite(U.u1_norm(m))
        @test U.u1_norm(m) ≈ 1.
    end
end
