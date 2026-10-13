# Benchmark adapters frozen from the original SA learning diagnostics.
# No dependency on ignored pilot scripts or external repositories.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using Optim, OhMyU1, Random, LinearAlgebra, Statistics, SHA, JLD2, SparseArrays, Printf
using OhMyU1: OptimizationProblem, assignment_matrix, fill_assignment_vectors!,
    build_mps_from_feasible_samples, ChargeIndex, init_u1_mps, sample_nondeg!,
    u1_norm, getval, train_nondeg!, TrainParams, orthogonalize!, normalize!,
    ExactPartitionSampler, fill_partition_vectors!
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("Use one Julia compute thread")
include("structural.jl")
"""Exact x'Qx for binary vectors, including nonsymmetric Q and diagonal terms.

Both methods call this same black-box evaluator; neither receives cost deltas.
"""
struct BinaryQuadratic
    Q::Matrix{Float64}
    active::Vector{Int}
end
function (f::BinaryQuadratic)(x)
    empty!(f.active)
    @inbounds for i in eachindex(x)
        x[i] == 1 && push!(f.active, i)
    end
    value = 0.0
    @inbounds for j in f.active, i in f.active
        value += f.Q[i,j]
    end
    value
end

"""Uniform random transposition of two rows' assigned columns.

The symmetric proposal preserves all row/column sums and connects S_n.
Optim stores floating-point states; all entries remain exactly zero or one.
"""
function assignment_neighbor!(current, proposed, n, rng)
    copyto!(proposed, current)
    i = rand(rng, 1:n); j = rand(rng, 1:n-1); j += j >= i
    a = findfirst(==(1), view(current, (i-1)*n+1:i*n))
    b = findfirst(==(1), view(current, (j-1)*n+1:j*n))
    proposed[(i-1)*n+a] = 0; proposed[(j-1)*n+b] = 0
    proposed[(i-1)*n+b] = 1; proposed[(j-1)*n+a] = 1
    nothing
end

function sa_trial(input,budget;restarts=10,max_steps_per_chain=typemax(Int), objective=BinaryQuadratic(input.Q,Int[]),
                  A=assignment_matrix(input.n), neighbor=nothing, validate=x->all(A*x .== 1))
    n=input.n; Random.seed!(input.seed+1000000)
    rng=MersenneTwister(input.seed+2000000)
    start=time_ns(); elapsed()=(time_ns()-start)/1e9
    costs=[objective(x) for x in eachcol(input.initial)]
    order=sortperm(costs); incumbent=minimum(costs); initial_best=incumbent
    solution=copy(input.initial[:,first(order)])
    unique_initial=unique(input.initial[:,order];dims=2)
    temperature=max(std(costs),eps(Float64))
    evaluations=length(costs); steps=0; completed=0
    history=[(elapsed(),incumbent)]; evaluation_history=[(evaluations,incumbent)]
    initialization=elapsed()
    # Ten independent chains seeded by the ten best distinct observed points.
    # A geometric wall-time schedule cools from std(initial costs) to 0.001 of it.
    for restart in 1:restarts
        elapsed()>=budget && break
        chain_start=elapsed(); chain_end=budget*restart/restarts
        chain_duration=max(chain_end-chain_start,eps(Float64))
        function measured_cost(x)
            value=objective(x); evaluations+=1
            if elapsed()<=budget && value<incumbent
                incumbent=value; solution=Int.(x);push!(history,(elapsed(),incumbent));push!(evaluation_history,(evaluations,incumbent))
            end
            value
        end
        cooling(k)=temperature*exp(log(0.001)*clamp((elapsed()-chain_start)/chain_duration,0.0,1.0))
        neighbor!(x,y)=isnothing(neighbor) ? assignment_neighbor!(x,y,n,rng) : neighbor(x,y,rng)
        method=Optim.SimulatedAnnealing(neighbor=neighbor!,temperature=cooling)
        options=Optim.Options(iterations=max_steps_per_chain,time_limit=chain_duration,
            callback=state->elapsed()>=chain_end,store_trace=false,show_trace=false)
        result=Optim.optimize(measured_cost,Float64.(unique_initial[:,mod1(restart,size(unique_initial,2))]),method,options)
        steps+=Optim.iterations(result);completed+=1
    end
    actual=elapsed()
    validate(solution) || error("Infeasible SA result")
    # Integer/float inputs may select different BLAS accumulation paths.
    isapprox(objective(solution),incumbent;rtol=1e-12,atol=1e-12) || error("SA incumbent mismatch")
    Dict("c_min"=>incumbent,"initial_c_min"=>initial_best,"elapsed_seconds"=>actual,
         "objective_calls"=>evaluations,"iterations"=>steps,"completed_iterations"=>completed,
         "evaluation_history"=>evaluation_history,"generated"=>steps,"solution"=>solution,"history"=>history,"initialization_seconds"=>initialization)
end

# Prespecified wall-time cooling: initial scale to ratio*initial scale.
# Infinite duration is used only by compilation warmups.
function geometric_temperature(initial,elapsed,duration,ratio)
    initial*exp(log(ratio)*clamp(elapsed/max(duration,eps(Float64)),0.,1.))
end

function u1_trial(input,budget;full=false,backend=:legacy,maxiter=typemax(Int),samples_per_iteration=10000,
                  objective=BinaryQuadratic(input.Q,Int[]), A=assignment_matrix(input.n), full_factory=nothing, cooling=:adaptive, cooling_ratio=0.001)
    cooling in (:adaptive,:geometric) || error("Unknown cooling schedule")
    0<cooling_ratio<=1 || error("Cooling ratio must be in (0,1]")
    n=input.n;Random.seed!(input.seed+3000000)
    problem=OptimizationProblem(A=A,b=ones(Int,size(A,1)),cost_function=x->objective(x))
    start=time_ns();elapsed()=(time_ns()-start)/1e9;expired()=elapsed()>=budget
    calls=0
    measured(x)=(calls+=1;objective(x))
    T=unique(input.initial;dims=2);costs=[measured(x) for x in eachcol(T)]
    idx=sortperm(costs);T=T[:,idx];costs=costs[idx]
    incumbent=minimum(costs);initial_best=incumbent;solution=copy(T[:,1])
    history=[(elapsed(),incumbent)]; evaluation_history=[(calls,incumbent)]
    phases=Dict("build"=>0.0,"train"=>0.0,"sample"=>0.0); iteration_log=Any[]
    # The full topology is constructed once inside the budget. Each outer
    # iteration resets amplitudes (as the data-built EELS solver also does).
    template=full ? (isnothing(full_factory) ? full_indicator(n) : full_factory()) : nothing
    full && isnothing(full_factory) && canonical_full!(template,n)
    initialization=elapsed();iterations=0;completed=0;generated=0
    initial_temperature=max(length(costs)>1 ? std(costs) : 0.0,eps(Float64))
    cooling_start=NaN;temperature_log=Any[]
    while !expired() && iterations<maxiter
        iterations+=1
        tick=time_ns()
        mps=full ? deepcopy(template) : build_mps_from_feasible_samples(problem,T,1,true)
        phases["build"]+=(time_ns()-tick)/1e9
        expired() && break
        if !full;orthogonalize!(mps);normalize!(mps);end
        expired() && break
        spread=length(costs)>1 ? std(costs) : 0.0
        isnan(cooling_start) && (cooling_start=elapsed())
        temperature=cooling==:adaptive ? spread : geometric_temperature(initial_temperature,elapsed()-cooling_start,budget-cooling_start,cooling_ratio)
        weights=temperature>0 ? exp.(-(costs.-minimum(costs))./temperature) : ones(length(costs))
        weights./=sum(weights)
        push!(temperature_log,(iteration=iterations,seconds=elapsed(),temperature=temperature,spread=spread,ess=inv(sum(abs2,weights))))
        tick=time_ns()
        train_nondeg!(mps,1,T,weights,TrainParams(0.05,1));normalize!(mps)
        phases["train"]+=(time_ns()-tick)/1e9
        expired() && break
        batch=zeros(Int,size(T,1),100);accepted=Matrix{Int}(undef,size(T,1),samples_per_iteration)
        accepted_costs=Float64[];used=0
        for _ in 1:div(samples_per_iteration,100)
            expired() && break
            tick=time_ns()
            sample_nondeg!(mps,batch,100;backend)
            phases["sample"]+=(time_ns()-tick)/1e9
            expired() && break
            all(problem.A*batch .== 1) || error("Infeasible U1 sample")
            for x in eachcol(batch)
                expired() && break
                value=measured(x)
                expired() && break
                used+=1;generated+=1;accepted[:,used].=x;push!(accepted_costs,value)
                if value<incumbent
                    incumbent=value;solution=copy(x);push!(history,(elapsed(),incumbent));push!(evaluation_history,(calls,incumbent))
                end
            end
        end
        expired() && break
        pool=hcat(T,accepted[:,1:used]);poolcost=vcat(costs,accepted_costs)
        # Deduplicate by observed assignment, then retain the 400 best points.
        seen=Set{Tuple}();keep=Int[]
        for j in sortperm(poolcost)
            key=Tuple(pool[:,j]);key in seen && continue
            push!(seen,key);push!(keep,j);length(keep)==400 && break
        end
        T=pool[:,keep];costs=poolcost[keep];completed+=1
        push!(iteration_log,(iteration=completed,seconds=elapsed(),calls=calls,best=incumbent,unique_candidates=length(unique(Tuple(x) for x in eachcol(accepted[:,1:used])))))
    end
    actual=elapsed()
    all(problem.A*solution .== 1) || error("Infeasible U1 result")
    isapprox(objective(solution),incumbent;rtol=1e-12,atol=1e-12) || error("U1 incumbent mismatch")
    Dict("c_min"=>incumbent,"initial_c_min"=>initial_best,"elapsed_seconds"=>actual,
         "objective_calls"=>calls,"iterations"=>iterations,"completed_iterations"=>completed,
         "cooling"=>string(cooling),"cooling_ratio"=>cooling_ratio,"temperature_log"=>temperature_log,"evaluation_history"=>evaluation_history,"phases"=>phases,"iteration_log"=>iteration_log,"generated"=>generated,"solution"=>solution,"history"=>history,"initialization_seconds"=>initialization)
end
