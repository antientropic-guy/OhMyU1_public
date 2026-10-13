# Merge completed shards and export convenient tables; no optimization.
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","..")))
using JLD2, SparseArrays, SHA
root=normpath(joinpath(@__DIR__,"..",".."))
folder=isempty(ARGS) ? joinpath(root,"data","sa_cluster_v1","results") : abspath(only(ARGS))
combined=Dict{Tuple{String,Int,Int,String,Int,String},Any}()
protocols=Dict{String,Any}()
reference=nothing
for (family,n) in (("assignment",12),("assignment",14),("partition",12),("partition",14))
    path=joinpath(folder,"$(family)_n$(n).jld2")
    d=load(path);p=d["protocol"]
    @assert p.family==family && p.n==n
    signature=(p.manifest_sha256,p.sources,p.budgets,p.count,p.julia_version,p.optim_version)
    if isnothing(reference);global reference=signature;else;@assert reference==signature;end
    expected=p.count*length(p.objectives)*length(p.budgets)*length(p.methods)
    length(d["results"])==expected || error("Incomplete shard $path: $(length(d["results"]))/$(expected)")
    @assert Set(keys(d["results"]))==Set((i,o,b,m) for i in 1:p.count for o in p.objectives for b in p.budgets for m in p.methods)
    protocols["$(family)_n$(n)"]=p
    for ((instance,objective,budget,method),r) in d["results"]
        combined[(family,n,instance,objective,budget,method)]=r
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
