# Generic binary simulated annealing with a scalar quadratic penalty.
# Include after run.jl. No constraint-preserving move catalogue is used.
function penalty_parameters(costs,A)
    # Uses only the initial objective values, not objective coefficients or optima.
    span=maximum(costs)-minimum(costs)
    scale=max(span,std(costs),1e-8*max(1.,maximum(abs,costs)))
    alpha=10scale
    # From a feasible binary point, one flip incurs ||A[:,j]||^2.
    violations=[sum(abs2,view(A,:,j)) for j in axes(A,2)]
    positive=filter(>(0),violations)
    typical=isempty(positive) ? 1. : median(positive)
    t0=max(std(costs),alpha*typical/log(2))
    tend=max(.001*std(costs),1e-10*scale)
    (;alpha,initial_temperature=t0,final_temperature=tend,initial_cost_span=span,
      typical_flip_violation=typical,scale)
end

function bitflip_neighbor!(current,proposed,rng)
    copyto!(proposed,current)
    j=rand(rng,eachindex(current));proposed[j]=1-current[j]
    nothing
end

function penalty_trial(record,budget;objective,restarts=10,max_steps_per_chain=typemax(Int))
    Random.seed!(record.seed+1000000)
    rng=MersenneTwister(record.seed+2000000)
    start=time_ns();elapsed()=(time_ns()-start)/1e9
    costs=[objective(x) for x in eachcol(record.initial)]
    cfg=penalty_parameters(costs,record.A)
    order=sortperm(costs);initial_best=minimum(costs);incumbent=initial_best
    starts=unique(record.initial[:,order];dims=2);solution=copy(starts[:,1])
    evaluations=length(costs);feasible_calls=evaluations;infeasible_calls=0
    penalty_queries=0;infeasible_beats_initial=0;min_infeasible_penalized=Inf
    best_penalized=incumbent;best_penalized_violation=0.;max_violation=0.;violation_sum=0.
    steps=0;completed=0;proposals=0
    history=[(elapsed(),incumbent)];evaluation_history=[(evaluations,incumbent)]
    penalized_history=[(elapsed(),best_penalized,0.)]
    call_checkpoints=[(elapsed(),evaluations,feasible_calls,infeasible_calls)]
    initialization=elapsed();residual=zeros(Float64,length(record.b))
    function measured(x)
        value=objective(x);evaluations+=1
        mul!(residual,record.A,x)
        violation=0.
        @inbounds for j in eachindex(residual)
            residual[j]-=record.b[j];violation+=abs2(residual[j])
        end
        penalty_queries+=1;max_violation=max(max_violation,violation);violation_sum+=violation
        penalized=value+cfg.alpha*violation
        if iszero(violation)
            feasible_calls+=1
            if elapsed()<=budget && value<incumbent
                incumbent=value;solution=Int.(x)
                push!(history,(elapsed(),incumbent));push!(evaluation_history,(evaluations,incumbent))
            end
        else
            infeasible_calls+=1
            min_infeasible_penalized=min(min_infeasible_penalized,penalized)
            infeasible_beats_initial+=penalized<initial_best
        end
        if elapsed()<=budget && penalized<best_penalized
            best_penalized=penalized;best_penalized_violation=violation
            push!(penalized_history,(elapsed(),penalized,violation))
        end
        # Exact counters at periodic evaluation checkpoints; no extra oracle calls.
        if evaluations%10000==0
            push!(call_checkpoints,(elapsed(),evaluations,feasible_calls,infeasible_calls))
        end
        penalized
    end
    function neighbor!(x,y)
        proposals+=1;bitflip_neighbor!(x,y,rng)
    end
    for chain in 1:restarts
        elapsed()>=budget && break
        chain_start=elapsed();chain_end=budget*chain/restarts
        duration=max(chain_end-chain_start,eps(Float64))
        cooling(k)=cfg.initial_temperature*exp(log(cfg.final_temperature/cfg.initial_temperature)*clamp((elapsed()-chain_start)/duration,0.,1.))
        method=Optim.SimulatedAnnealing(neighbor=neighbor!,temperature=cooling)
        options=Optim.Options(iterations=max_steps_per_chain,time_limit=duration,
            callback=state->elapsed()>=chain_end,store_trace=false,show_trace=false)
        answer=Optim.optimize(measured,Float64.(starts[:,mod1(chain,size(starts,2))]),method,options)
        steps+=Optim.iterations(answer);completed+=1
        push!(call_checkpoints,(elapsed(),evaluations,feasible_calls,infeasible_calls))
    end
    actual=elapsed()
    push!(call_checkpoints,(actual,evaluations,feasible_calls,infeasible_calls))
    @assert all(x->x in (0,1),solution) && record.A*solution==record.b
    @assert isapprox(objective(solution),incumbent;atol=1e-9,rtol=1e-12)
    @assert feasible_calls+infeasible_calls==evaluations
    Dict{String,Any}("c_min"=>incumbent,"initial_c_min"=>initial_best,"solution"=>solution,
        "elapsed_seconds"=>actual,"objective_calls"=>evaluations,"iterations"=>steps,
        "completed_iterations"=>completed,"generated"=>proposals,"history"=>history,
        "evaluation_history"=>evaluation_history,"initialization_seconds"=>initialization,
        "alpha"=>cfg.alpha,"initial_temperature"=>cfg.initial_temperature,
        "final_temperature"=>cfg.final_temperature,"initial_cost_span"=>cfg.initial_cost_span,
        "typical_flip_violation"=>cfg.typical_flip_violation,"feasible_calls"=>feasible_calls,
        "infeasible_calls"=>infeasible_calls,"penalty_queries"=>penalty_queries,
        "infeasible_beats_initial"=>infeasible_beats_initial,
        "min_infeasible_penalized"=>min_infeasible_penalized,"best_penalized"=>best_penalized,
        "best_penalized_violation"=>best_penalized_violation,"max_violation"=>max_violation,
        "mean_violation"=>violation_sum/max(penalty_queries,1),
        "penalized_history"=>penalized_history,"call_checkpoints"=>call_checkpoints)
end
