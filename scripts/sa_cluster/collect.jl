# Merge completed shards and export convenient tables; no optimization.
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","..")))
using JLD2, SparseArrays, SHA
root=normpath(joinpath(@__DIR__,"..",".."))
folder=isempty(ARGS) ? joinpath(root,"data","sa_cluster_v2","results") : abspath(only(ARGS))
combined=Dict{Tuple{String,Int,Int,String,Int,String},Any}()
protocols=Dict{String,Any}()
reference=nothing
for family in ("assignment","partition"), objective in ("quadratic","entropic")
    firstpath=joinpath(folder,"$(family)_$(objective)_worker1.jld2")
    firstprotocol=load(firstpath,"protocol")
    for worker in 1:firstprotocol.workers
    path=joinpath(folder,"$(family)_$(objective)_worker$(worker).jld2")
    d=load(path);p=d["protocol"]
    @assert p.version==2 && p.family==family && p.objectives==[objective] && p.worker==worker
    signature=(p.manifest_sha256,p.sources,p.budgets,p.count,p.sizes,p.workers,p.methods,p.julia_version,p.optim_version)
    if isnothing(reference);global reference=signature;else;@assert reference==signature;end
    assigned=[(n,i) for i in 1:p.count for n in sort(p.sizes)][worker:p.workers:end]
    @assert p.assigned==assigned
    expected=length(assigned)*length(p.objectives)*length(p.budgets)*length(p.methods)
    length(d["results"])==expected || error("Incomplete shard $path: $(length(d["results"]))/$(expected)")
    @assert Set(keys(d["results"]))==Set((n,i,o,b,m) for (n,i) in assigned for o in p.objectives for b in p.budgets for m in p.methods)
    protocols["$(family)_$(objective)_worker$(worker)"]=p
    for ((n,instance,objective,budget,method),r) in d["results"]
        @assert !haskey(combined,(family,n,instance,objective,budget,method))
        combined[(family,n,instance,objective,budget,method)]=r
    end
    end
end
temporary=joinpath(folder,"combined.jld2.pending")
jldsave(temporary,true,IOStream;results=combined,protocols)
mv(temporary,joinpath(folder,"combined.jld2");force=true)
open(joinpath(folder,"summary.csv"),"w") do io
    println(io,"family,n,instance,objective,budget,method,initial,final,seconds,calls,iterations,initialization_s,build_s,train_s,sample_s,allocated_bytes,gc_seconds,process_maxrss_bytes")
    for key in sort(collect(keys(combined)))
        r=combined[key];phases=get(r,"phases",Dict())
        println(io,join((key...,r["initial_c_min"],r["c_min"],r["elapsed_seconds"],r["objective_calls"],r["completed_iterations"],
            r["initialization_seconds"],get(phases,"build",0.),get(phases,"train",0.),get(phases,"sample",0.),
            r["allocated_bytes"],r["gc_seconds"],r["process_maxrss_bytes"]),','))
    end
end
open(joinpath(folder,"curves.csv"),"w") do io
    println(io,"family,n,instance,objective,budget,method,seconds,calls,best")
    for key in sort(collect(keys(combined)))
        r=combined[key]
        for ((seconds,best),(calls,best2)) in zip(r["history"],r["evaluation_history"])
            @assert best==best2
            println(io,join((key...,seconds,calls,best),','))
        end
    end
end
println("Collected $(length(combined)) runs into combined.jld2, summary.csv and curves.csv")
