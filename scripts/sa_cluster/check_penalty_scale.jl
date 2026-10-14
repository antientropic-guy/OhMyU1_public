# Check penalty calibration on frozen initial data and generic binary probes.
# This performs no SA/TT optimization and selects no hyperparameters by outcomes.
include("run_penalty.jl")
out=joinpath(ROOT,"data","sa_penalty_v1");mkpath(out)
manifest=load(joinpath(ROOT,"data","sa_cluster_v2","manifest.jld2"),"manifest")
rows=NamedTuple[]
for entry in manifest.records
    record=load(joinpath(ROOT,"data","sa_cluster_v2","inputs",entry.file),"record")
    for o in ("quadratic","entropic")
        objective=make_objective(record,o)
        costs=[objective(x) for x in eachcol(record.initial)]
        params=penalty_parameters(costs,record.A)
        best=record.initial[:,argmin(costs)];U=minimum(costs)
        probes=[copy(best) for _ in eachindex(best)]
        for j in eachindex(best);probes[j][j]=1-best[j];end
        rng=MersenneTwister(record.seed+4000000)
        append!(probes,[rand(rng,0:1,length(best)) for _ in 1:32])
        for _ in 1:64
            x=copy(best)
            for j in randperm(rng,length(x))[1:rand(rng,2:min(8,length(x)))];x[j]=1-x[j];end
            push!(probes,x)
        end
        margins=Float64[]
        for x in probes
            v=sum(abs2,record.A*x-record.b)
            v==0 && continue
            push!(margins,objective(x)+params.alpha*v-U)
        end
        push!(rows,(family=entry.family,objective=o,n=entry.n,instance=entry.instance,
            alpha=params.alpha,T0=params.initial_temperature,Tend=params.final_temperature,
            initial_span=params.initial_cost_span,infeasible_probes=length(margins),
            probes_beating_initial=count(<(0),margins),minimum_penalized_margin=minimum(margins)))
    end
end
function csv_file(path,rows)
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows;println(io,join(values(r),','));end
    end
end
csv_file(joinpath(out,"calibration.csv"),rows)
summary=NamedTuple[]
for f in ("assignment","partition"),o in ("quadratic","entropic"),n in (10,12,14)
    g=filter(r->r.family==f && r.objective==o && r.n==n,rows)
    push!(summary,(family=f,objective=o,n=n,alpha_min=minimum(r.alpha for r in g),
        alpha_median=median([r.alpha for r in g]),alpha_max=maximum(r.alpha for r in g),
        T0_median=median([r.T0 for r in g]),Tend_median=median([r.Tend for r in g]),
        infeasible_probes=sum(r.infeasible_probes for r in g),probes_beating_initial=sum(r.probes_beating_initial for r in g)))
end
csv_file(joinpath(out,"calibration_summary.csv"),summary)
println("Checked $(length(rows)) input/objective combinations, $(sum(r.infeasible_probes for r in rows)) infeasible probes; $(sum(r.probes_beating_initial for r in rows)) beat the initial feasible best under the penalty.")
