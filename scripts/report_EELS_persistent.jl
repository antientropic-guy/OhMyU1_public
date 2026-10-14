# Audit and paired A/B report for the frozen protocol-3 assignment results.
# No optimization is executed. Shared statistics match the ADD reports.
include(joinpath(@__DIR__,"report_ADD_tuning.jl"))
using Random
ENV["GKSwstype"]="100"
using StatsPlots
BLAS.set_num_threads(1)
root=normpath(joinpath(@__DIR__,".."))
input=joinpath(root,"data","eels_optimization","gamma1_persistent_v3")
out=joinpath(root,"examples","eels_persistent_analysis");mkpath(out)
ns=12:16;ids=1:50
rows=NamedTuple[];summaries=NamedTuple[];by_size=NamedTuple[];audit=String[]
for objective in ("linear","quadratic")
    path=joinpath(input,"res_EELS_gamma1_$(objective)_test.jld2")
    d=load(path);cfg=d["configuration"];sp=d["solver_params"]
    @assert cfg.protocol==3 && cfg.objective==objective && cfg.gamma==1 && cfg.split=="test"
    @assert cfg.ns==collect(ns) && cfg.instances==collect(ids)
    @assert cfg.external_samples_after_initialization==0 && cfg.sampling_backend=="legacy"
    @assert cfg.model_policy=="initialize once; sample then train; retain amplitudes"
    @assert cfg.selection=="best NUM_FEASIBLE_SAMPLES distinct points from current batch only"
    @assert sp.NUM_FEASIBLE_SAMPLES==400 && sp.NUM_MPS_SAMPLES==10000 && sp.NUM_GLOBAL_ITER==20
    @assert sp.NUM_SWEEP_ITER==1 && sp.LEARNING_RATE==.05 && sp.LINK_DEGENERACY==1
    @assert Set(keys(d["res_dict"]))==Set(["best_cost","EELS"])
    for arm in ("best_cost","EELS")
        @assert Set(keys(d["res_dict"][arm]))==Set(ns)
        @assert Set(keys(d["diagnostics"][arm]))==Set((n,i) for n in ns for i in ids)
        for n in ns;@assert Set(keys(d["res_dict"][arm][n]))==Set(ids);end
    end
    for n in ns,i in ids
        filename=objective=="linear" ? "c_$(n)_$(i)_r=50_test.jld2" : "q_$(n)_$(i)_r=7_test.jld2"
        source=joinpath(root,"data",objective=="linear" ? "test_random_vectors" : "test_random_bilinear_forms",filename)
        @assert bytes2hex(sha256(read(source)))==cfg.hashes[(n,i)]
        coeff=load(source,objective=="linear" ? "c" : "q")
        objective_cost(x)=objective=="linear" ? dot(coeff,x) : dot(x,coeff*x)
        # Reproduce the initial dataset, including sorting and column order.
        seed=20261006+1000000+100000*(objective=="quadratic")+100n+i
        Random.seed!(seed)
        initial=zeros(Int,n*n,400);OhMyU1.fill_assignment_vectors!(initial,n)
        initial=unique(initial;dims=2);initial_costs=[objective_cost(x) for x in eachcol(initial)]
        initial=initial[:,sortperm(initial_costs)]
        io=IOBuffer();print(io,size(initial),':');write(io,UInt8.(vec(initial)))
        fingerprint=bytes2hex(sha256(take!(io)))
        base=d["res_dict"]["best_cost"][n][i];eels=d["res_dict"]["EELS"][n][i]
        for arm in ("best_cost","EELS")
            r=d["res_dict"][arm][n][i];info=d["diagnostics"][arm][(n,i)]
            @assert info.initial_training_sha256==fingerprint
            @assert info.model_initializations==1 && info.eels_applications==Int(arm=="EELS")
            @assert length(r.learning_curve)==21 && last(r.learning_curve)==r.c_min
            @assert first(r.learning_curve)==minimum(initial_costs)
            @assert all(diff(r.learning_curve).<=1e-9)
            @assert length(info.training_sizes)==20 && info.training_sizes==info.elite_sizes
            @assert all(s->1<=s<=400,info.training_sizes)
            @assert length(info.support_sizes)==20 && all(==(first(info.support_sizes)),info.support_sizes)
            @assert length(info.evaluations)==21 && issorted(info.evaluations)
            @assert length(info.elapsed_seconds)==21 && issorted(info.elapsed_seconds)
            @assert all(x->x in (0,1),r.incub) && all(OhMyU1.assignment_matrix(n)*r.incub .== 1)
            @assert isapprox(objective_cost(r.incub),r.c_min;atol=1e-8,rtol=1e-12)
        end
        bd=d["diagnostics"]["best_cost"][(n,i)];ed=d["diagnostics"]["EELS"][(n,i)]
        @assert first(ed.support_sizes)>=first(bd.support_sizes)*(1-1e-10)
        push!(rows,(;objective,n,variables=n*n,instance=i,baseline=base.c_min,eels=eels.c_min,
            improvement_pct=100*(base.c_min-eels.c_min)/abs(base.c_min),initial=first(base.learning_curve),
            baseline_unique_evaluations=last(bd.evaluations),eels_unique_evaluations=last(ed.evaluations),
            baseline_seconds=last(bd.elapsed_seconds),eels_seconds=last(ed.elapsed_seconds),
            baseline_support=first(bd.support_sizes),eels_support=first(ed.support_sizes)))
    end
    group=filter(r->r.objective==objective,rows)
    push!(summaries,merge((;objective),paired_summary([r.baseline for r in group],[r.eels for r in group])))
    for n in ns
        g=filter(r->r.n==n,group)
        push!(by_size,merge((;objective,n,variables=n*n),paired_summary([r.baseline for r in g],[r.eels for r in g])))
    end
    push!(audit,"$objective: 250 complete pairs; SHA256 $(bytes2hex(sha256(read(path)))); Julia $(cfg.julia_version).")
end
adjusted=holm_adjust([r.p for r in summaries])
summaries=[merge(r,(;p_holm2=adjusted[j])) for (j,r) in enumerate(summaries)]
adjusted_groups=holm_adjust([r.p for r in by_size])
by_size=[merge(r,(;p_holm10=adjusted_groups[j])) for (j,r) in enumerate(by_size)]
write_csv(joinpath(out,"paired_costs.csv"),rows)
write_csv(joinpath(out,"summary.csv"),summaries)
write_csv(joinpath(out,"by_size.csv"),by_size)
open(joinpath(out,"audit.md"),"w") do io
    println(io,"Validated 500 paired tasks / 1,000 runs. Both protocol-3 files are complete. All objective-file hashes match; all final solutions are binary, feasible, and have the stored cost. Initial samples were regenerated from the saved seed rule, sorted as in the runner, and matched both training-data fingerprints exactly. Histories are monotone; each arm records exactly one initialization and 20 iterations; EELS is applied only once in the enhanced arm. Training sizes never exceed 400. No external samples are injected.\n")
    foreach(s->println(io,s,"\n"),audit)
end
lo=minimum(r.improvement_pct for r in rows);hi=maximum(r.improvement_pct for r in rows);padding=.05*max(hi-lo,1.)
for objective in ("linear","quadratic")
    g=filter(r->r.objective==objective,rows)
    p=boxplot([r.variables for r in g],[r.improvement_pct for r in g];
        title="$(uppercasefirst(objective)) assignment: persistent TT, EELS vs Best Cost",
        xlabel="Number of binary variables",ylabel="Improvement over Best Cost (%)",label="",
        color=:steelblue,alpha=.5,whisker_range=1.5,outliers=true,bar_width=12,
        xticks=collect(ns).^2,ylims=(lo-padding,hi+padding),yformatter=x->@sprintf("%.1f",x),
        legend=:topleft,size=(850,450),margin=5Plots.mm)
    means=[mean(r.improvement_pct for r in g if r.n==n) for n in ns]
    medians=[median([r.improvement_pct for r in g if r.n==n]) for n in ns]
    plot!(p,collect(ns).^2,means;color=:darkorange,marker=:diamond,label="Group average",lw=2)
    plot!(p,collect(ns).^2,medians;color=:purple,marker=:circle,label="Group median",lw=2)
    hline!(p,[0.];color=:gray,linestyle=:dash,label="")
    savefig(p,joinpath(out,"EELS_$(objective).png"))
    savefig(p,joinpath(out,"EELS_$(objective).svg"))
end
foreach(println,summaries)
println("Saved ",out)
