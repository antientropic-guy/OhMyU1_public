# Collect all 128 private shards. Never execute optimization.
using JLD2, SparseArrays
root=normpath(joinpath(@__DIR__,"..",".."))
folder=isempty(ARGS) ? joinpath(root,"data","sa_penalty_v1","results") : abspath(only(ARGS))
combined=Dict{Tuple{String,Int,Int,String,Int,String},Any}();protocols=Dict{String,Any}()
reference=nothing
for family in ("assignment","partition"),objective in ("quadratic","entropic")
    firstprotocol=load(joinpath(folder,"$(family)_$(objective)_worker1.jld2"),"protocol")
    for worker in 1:firstprotocol.workers
        path=joinpath(folder,"$(family)_$(objective)_worker$(worker).jld2")
        d=load(path);p=d["protocol"]
        @assert p.version==1 && p.family==family && p.objectives==[objective] && p.worker==worker && p.methods==["SA_penalty"]
        signature=(p.manifest_sha256,p.sources,p.budgets,p.count,p.sizes,p.workers,p.alpha_rule,p.temperature,p.proposal,p.julia_version,p.optim_version)
        if isnothing(reference);global reference=signature;else;@assert reference==signature;end
        assigned=[(n,i) for i in 1:p.count for n in sort(p.sizes)][worker:p.workers:end]
        @assert p.assigned==assigned
        expected=Set((n,i,objective,b,"SA_penalty") for (n,i) in assigned for b in p.budgets)
        Set(keys(d["results"]))==expected || error("Incomplete or unexpected shard $path")
        protocols["$(family)_$(objective)_worker$(worker)"]=p
        for ((n,i,o,b,m),r) in d["results"]
            key=(family,n,i,o,b,m);@assert !haskey(combined,key);combined[key]=r
        end
    end
end
temporary=joinpath(folder,"combined.jld2.pending")
jldsave(temporary,true,IOStream;results=combined,protocols)
mv(temporary,joinpath(folder,"combined.jld2");force=true)
open(joinpath(folder,"summary.csv"),"w") do io
    println(io,"family,n,instance,objective,budget,method,initial,final,seconds,calls,chains,alpha,initial_temperature,final_temperature,feasible_calls,infeasible_calls,best_penalized,best_penalized_violation,infeasible_beats_initial,max_violation")
    for key in sort(collect(keys(combined)))
        r=combined[key]
        fields=["initial_c_min","c_min","elapsed_seconds","objective_calls","completed_iterations","alpha","initial_temperature","final_temperature","feasible_calls","infeasible_calls","best_penalized","best_penalized_violation","infeasible_beats_initial","max_violation"]
        println(io,join((key...,[r[k] for k in fields]...),','))
    end
end
open(joinpath(folder,"curves.csv"),"w") do io
    println(io,"family,n,instance,objective,budget,method,seconds,calls,best")
    for key in sort(collect(keys(combined)))
        r=combined[key]
        for ((t,v),(c,w)) in zip(r["history"],r["evaluation_history"])
            @assert v==w;println(io,join((key...,t,c,v),','))
        end
    end
end
open(joinpath(folder,"call_checkpoints.csv"),"w") do io
    println(io,"family,n,instance,objective,budget,method,seconds,calls,feasible_calls,infeasible_calls")
    for key in sort(collect(keys(combined))),row in combined[key]["call_checkpoints"]
        println(io,join((key...,row...),','))
    end
end
println("Collected $(length(combined)) penalty-SA runs.")
