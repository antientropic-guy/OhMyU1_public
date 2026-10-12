# Moderate fixed assignment, real public solve(), three full iterations.
# The initial set excludes the identity optimum; the next external batch
# contains it. Its retention forces the exact answer from iteration two.
function benchmark_problem()
    n=5;A=U.assignment_matrix(n);b=ones(Int,2n)
    perms=Vector{Int}[]
    function visit(p,available)
        if isempty(available);push!(perms,copy(p));return;end
        for v in available;visit(vcat(p,v),filter(!=(v),available));end
    end
    visit(Int[],collect(1:n))
    covers=zeros(Int,n*n,length(perms))
    for (k,p) in enumerate(perms),j in 1:n;covers[(j-1)*n+p[j],k]=1;end
    c=Float64[i==j ? 0 : 1+abs(i-j)/10 for i in 1:n for j in 1:n]
    problem(A,b;cost=x->dot(c,x)),covers
end
function benchmark_run(parallel)
    opt,covers=benchmark_problem();calls=Ref(0)
    function source!(X)
        calls[]+=1
        for j in axes(X,2)
            k=calls[]==1 ? 2+mod(j-1,size(covers,2)-1) : 1+mod(j-1,size(covers,2))
            X[:,j].=view(covers,:,k)
        end
    end
    sp=U.SolverParams(NUM_FEASIBLE_SAMPLES=32,NUM_MPS_SAMPLES=512,NUM_GLOBAL_ITER=3,
        NUM_SWEEP_ITER=1,UTILITY_FRACTION=.125,KEEP_NUM_WORST=0.)
    Random.seed!(6021)
    stats=U.solve(opt,source!,sp,"best_cost";parallel,print_stats=false)
    @test opt.A*stats.incub==opt.b
    @test stats.c_min==0.
    @test stats.learning_curve[end]==0 && all(diff(stats.learning_curve).<=0)
    @test length(stats.learning_curve)==4 && calls[]==4
    stats
end
function performance_tests()
    result=Dict{String,Any}()
    for parallel in (false,true)
        label=parallel ? "parallel" : "serial"
        @testset "End-to-end $label" begin
            benchmark_run(parallel) # exclude first-use compilation
            times=Float64[];bytes=Int[];gcs=Float64[]
            for _ in 1:3
                GC.gc();r=@timed benchmark_run(parallel)
                push!(times,r.time);push!(bytes,r.bytes);push!(gcs,r.gctime)
            end
            result[label]=Dict("seconds"=>median(times),"allocated_bytes"=>median(bytes),"gc_seconds"=>median(gcs),"cost"=>0.)
            # Portable smoke ceilings; tighter same-machine limits use an explicit baseline.
            @test median(times)<8.
            @test median(bytes)<350_000_000
        end
    end
    result["julia"]=string(VERSION);result["cpu"]=Sys.CPU_NAME
    result["threads"]=Threads.nthreads();result["interactive_threads"]=Threads.nthreads(:interactive)
    result["max_rss_bytes"]=Sys.maxrss()
    @test result["max_rss_bytes"]<2_500_000_000
    result
end
