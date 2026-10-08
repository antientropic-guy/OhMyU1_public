using OhMyU1,Random,LinearAlgebra,Statistics,SHA
using OhMyU1: OptimizationProblem,assignment_matrix,build_mps_from_feasible_samples,
    orthogonalize!,normalize!,u1_norm,train_nondeg!,TrainParams,sample_nondeg!
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("Use --threads=1")

function trial(n,objective,rep,budget;eels=true,maxiter=typemax(Int))
    root=normpath(joinpath(@__DIR__,"..",".."))
    stem=joinpath(root,"data","protes_comparison","scaling","paired_inputs","$(objective)_$(n)_$(rep)")
    raw=read(stem*".i32");value_raw=read(stem*".f64");hashes=readlines(stem*".sha256")
    bytes2hex(sha256(raw))==hashes[1] || error("Initial hash mismatch")
    bytes2hex(sha256(value_raw))==hashes[2] || error("Objective hash mismatch")
    initial=Int.(reshape(reinterpret(Int32,raw),n*n,400))
    val=copy(reinterpret(Float64,value_raw));value=objective=="linear" ? val : reshape(val,n*n,n*n)
    cost=objective=="linear" ? x->dot(x,value) : x->dot(x,value*x)
    problem=OptimizationProblem(A=assignment_matrix(n),b=ones(Int,2n),cost_function=cost)
    seed=20261008+n*1000+rep*10+(objective=="quadratic")
    initial_set=Set(Tuple(x) for x in eachcol(initial));seen=Set{Tuple}()
    generated=0;iterations=0;completed=0;initial_fit=false
    cache=Dict{Tuple,Float64}()
    cached(x)=get!(cache,Tuple(x)) do; Float64(cost(x));end
    start=time_ns();expired()=(time_ns()-start)/1e9>=budget
    T=unique(initial;dims=2);costs=[cached(x) for x in eachcol(T)]
    order=sortperm(costs);T=T[:,order];costs=costs[order];incumbent=minimum(costs);initial_best=incumbent
    while !expired() && iterations<maxiter
        iterations+=1
        mps=build_mps_from_feasible_samples(problem,T,1,eels)
        expired() && break
        orthogonalize!(mps);normalize!(mps)
        expired() && break
        temperature=length(costs)>1 ? std(costs) : 0.0
        weights=temperature>0 ? exp.(-(costs.-minimum(costs))./temperature) : ones(length(costs))
        weights./=sum(weights)
        train_nondeg!(mps,1,T,weights,TrainParams(0.05,1));normalize!(mps)
        initial_fit=true
        expired() && break
        Random.seed!(seed+10_000_000iterations)
        accepted=Vector{Vector{Int}}();batch=zeros(Int,n*n,100)
        for _ in 1:100 # standard 10,000 samples per outer iteration, deadline checks every 100
            expired() && break
            sample_nondeg!(mps,batch,100)
            expired() && break # discard a batch arriving after the deadline, as for PROTES
            all(problem.A*batch .== 1) || error("Infeasible generated sample")
            generated+=100
            for x in eachcol(batch)
                key=Tuple(x);push!(seen,key);push!(accepted,copy(x));incumbent=min(incumbent,cached(x))
            end
        end
        expired() && break
        pool=unique(hcat(T,reduce(hcat,accepted));dims=2)
        poolcost=[cached(x) for x in eachcol(pool)];idx=sortperm(poolcost)[1:min(400,length(poolcost))]
        T=pool[:,idx];costs=poolcost[idx];completed+=1
    end
    elapsed=(time_ns()-start)/1e9
    (n,objective,rep,budget,eels ? "EELS" : "BestCost",generated,generated,length(seen),length(setdiff(seen,initial_set)),
     elapsed,iterations,completed,initial_fit,initial_best,incumbent,hashes[1],hashes[2])
end

function main()
    root=normpath(joinpath(@__DIR__,"..",".."))
    stage=isempty(ARGS) ? "short" : ARGS[1]
    stage in ("short","long") || error("Use short or long")
    out=joinpath(root,"data","protes_comparison","scaling","data_u1_$(stage).csv")
    # Same-dimension warmup on repetition 2, outside all measured budgets.
    open(out,"w") do io
        println(io,"n,objective,repetition,budget,method,generated,feasible,unique,novel_unique,elapsed_seconds,iterations,completed_iterations,initial_training_complete,initial_c_min,c_min,initial_sha256,objective_sha256")
        for n in (stage=="short" ? (4:10) : (8:8)), objective in ("linear","quadratic")
            trial(n,objective,2,Inf;maxiter=1)
            for rep in (stage=="short" ? (0:0) : (0:2))
                GC.gc();row=trial(n,objective,rep,stage=="short" ? 5.0 : 50.0)
                println(io,join(row,','));flush(io)
                println("Data U1 ",join(row[1:13],','));flush(stdout)
            end
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
