# Realistic assignment workload. The external source draws uniform permutations
# using its own RNG; model draws cannot advance the source RNG. The solver
# requests enough external points to fill its 20,000-candidate pool.
const LARGE_WORKLOAD = "assignment10_initial1000_samples5000_iterations10_sweeps2_seed8101_v1"
function large_benchmark_run(parallel; iterations=10, strategy="best_cost")
    n=10; rng=MersenneTwister(8101); source_rng=MersenneTwister(8102)
    A=U.assignment_matrix(n); b=ones(Int,2n); c=randn(rng,n*n)
    opt=problem(A,b;cost=x->dot(c,x)); calls=Ref(0)
    function source!(X)
        calls[]+=1
        fill!(X,0)
        for j in axes(X,2), (i,v) in enumerate(randperm(source_rng,n))
            X[(i-1)*n+v,j]=1
        end
        return nothing
    end
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=1000,NUM_MPS_SAMPLES=5000,
        NUM_GLOBAL_ITER=iterations,NUM_SWEEP_ITER=2,UTILITY_FRACTION=.05,KEEP_NUM_WORST=0.)
    Random.seed!(8103)
    stats=U.solve(opt,source!,sp,strategy;parallel,print_stats=false,mixed_proportion=.4)
    @test opt.A*stats.incub==opt.b
    @test all(x->x in (0,1),stats.incub)
    @test isfinite(stats.c_min) && stats.c_min ≈ opt.cost_function(stats.incub)
    @test length(stats.learning_curve)==iterations+1 && all(diff(stats.learning_curve).<=0)
    @test stats.learning_curve[end]==stats.c_min
    @test calls[]==iterations+1
    return stats
end

function large_performance_tests()
    parallel=Threads.nthreads()>1
    println("Assignment 10x10: warm-up, then three measured 10-iteration solves; parallel=$parallel")
    flush(stdout)
    warm_started=time()
    large_benchmark_run(parallel;iterations=1) # same shapes and code paths, JIT excluded
    warm_seconds=time()-warm_started
    times=Float64[]; bytes=Int[]; gcs=Float64[]; costs=Float64[]; initial=Float64[]
    for rep in 1:3
        GC.gc()
        r=@timed large_benchmark_run(parallel)
        push!(times,r.time);push!(bytes,r.bytes);push!(gcs,r.gctime)
        push!(costs,r.value.c_min);push!(initial,first(r.value.learning_curve))
        println("Measured run $rep: $(round(r.time;digits=3)) s, $(r.bytes) allocated bytes, cost=$(r.value.c_min)")
        flush(stdout)
    end
    rss=Sys.maxrss()
    @test median(times)<120.
    @test median(bytes)<15_000_000_000
    @test rss<3_000_000_000
    @test all(costs .< initial) # optimization must actually improve the initial set
    return Dict("workload"=>LARGE_WORKLOAD,"threads"=>Threads.nthreads(),
        "interactive_threads"=>Threads.nthreads(:interactive),"parallel_sampling"=>parallel,
        "seconds"=>median(times),"allocated_bytes"=>median(bytes),"gc_seconds"=>median(gcs),
        "max_rss_bytes"=>rss,"warmup_seconds"=>warm_seconds,
        "times"=>times,"allocations"=>bytes,"final_costs"=>costs,"initial_costs"=>initial)
end
