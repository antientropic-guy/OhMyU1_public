# Read-only scientific audit and plots for the completed cluster benchmark.
# Run collect.jl first. No optimizer, input generation, or cluster submission.
using JLD2, SparseArrays, LinearAlgebra, Statistics, SHA, Printf, HypothesisTests
ENV["GKSwstype"] = "100"
using Plots, StatsPlots
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = joinpath(ROOT,"data","sa_cluster_v2")
const OUT = joinpath(ROOT,"examples","sa_cluster_analysis")
mkpath(OUT)
const METHODS = ["SA","U1_data_EELS","U1_full"]
const LABELS = ["SA","Data + EELS","Full MPS"]
const COLORS = [:steelblue, :darkorange, :seagreen]
data = load(joinpath(INPUT,"results","combined.jld2"))
results = data["results"]; protocols=data["protocols"]
@assert length(results)==4320 && length(protocols)==128
manifest=load(joinpath(INPUT,"manifest.jld2"),"manifest")
records=Dict((e.family,e.n,e.instance)=>load(joinpath(INPUT,"inputs",e.file),"record") for e in manifest.records)
for e in manifest.records
    @assert bytes2hex(sha256(read(joinpath(INPUT,"inputs",e.file))))==e.sha256
end
manifest_hash=bytes2hex(sha256(read(joinpath(INPUT,"manifest.jld2"))))
for p in values(protocols)
    @assert p.manifest_sha256==manifest_hash
    @assert p.initial_samples==400 && p.mps_samples==10000 && p.elite_count==400
    @assert p.lr==0.05 && p.sweeps==1 && p.backend=="legacy" && p.sa_chains==10
    @assert p.mps_reset=="every iteration" && p.charge_degeneracy==1
end
function cost(record,x,kind)
    kind=="quadratic" && return dot(x,record.Q*x)
    o=record.entropic; z=o.scenarios*x/(o.tau*o.scale); peak=maximum(z)
    dot(o.c,x)/o.scale+o.tau*(peak+log(sum(exp.(z.-peak)))-log(length(z)))
end
initials=Dict((f,n,i,o)=>minimum(cost(r,x,o) for x in eachcol(r.initial)) for ((f,n,i),r) in records for o in ("quadratic","entropic"))
source_mismatches=String[]; newline_only=String[]
for (path,sha) in first(values(protocols)).sources
    if isfile(joinpath(ROOT,path)) && bytes2hex(sha256(read(joinpath(ROOT,path))))!=sha
        if bytes2hex(sha256(replace(read(joinpath(ROOT,path),String),"\r\n"=>"\n")))==sha
            push!(newline_only,path)
        else
            push!(source_mismatches,path)
        end
    end
end
hardware=Set{String}(); zero_initial=0
for ((family,n,i,o,b,m),r) in results
    rec=records[(family,n,i)];x=r["solution"]
    @assert all(v->v in (0,1),x) && rec.A*x==rec.b
    @assert isapprox(cost(rec,x,o),r["c_min"];atol=1e-9,rtol=1e-11)
    @assert isapprox(initials[(family,n,i,o)],r["initial_c_min"];atol=1e-9,rtol=1e-11)
    @assert bytes2hex(sha256(read(joinpath(INPUT,"inputs",r["input_file"]))))==r["input_sha256"]
    h=r["history"];ev=r["evaluation_history"]
    @assert length(h)==length(ev) && all(last.(h).==last.(ev))
    @assert issorted(first.(h)) && issorted(first.(ev)) && all(diff(last.(h)).<=0)
    @assert maximum(first.(h))<=b && maximum(first.(ev))<=r["objective_calls"]
    @assert last(h)[2]==r["c_min"] && r["elapsed_seconds"]>=last(h)[1]
    @assert r["hardware"].julia_threads==1 && r["hardware"].blas_threads==1
    push!(hardware,string(r["hardware"].cpu_model))
    @assert abs(r["initial_c_min"])>1e-10 && abs(r["c_min"])>1e-10
end
open(joinpath(OUT,"audit.md"),"w") do io
    println(io,"All 128 shards and 4,320 runs are complete. All 240 input hashes, paired initial costs, final binary feasibility, recomputed final objective values, monotone histories, deadline-filtered incumbents, and single-thread settings passed validation.")
    println(io,"\nHardware: ",join(sort(collect(hardware)),"; "),". Julia ",first(values(protocols)).julia_version,"; Optim ",first(values(protocols)).optim_version,". These are downloaded cluster runs, not local optimization runs.")
    println(io,"\nSaved source fingerprints differing from the current checkout: ",isempty(source_mismatches) ? "none" : join(sort(source_mismatches),", "),".")
    println(io,"\nFiles with only Windows/Linux line-ending differences: ",length(newline_only),". Line endings were normalized solely for this provenance comparison.")
end
function valat(r,t;calls=false)
    h=r[calls ? "evaluation_history" : "history"]
    k=searchsortedlast(first.(h),t)
    k==0 ? r["initial_c_min"] : h[k][2]
end
gain(r,v)=100*(r["initial_c_min"]-v)/abs(r["initial_c_min"])
groups=[(f,o,n,b) for f in ("assignment","partition") for o in ("quadratic","entropic") for n in (10,12,14) for b in (10,50,200)]
group(f,o,n,b,m)=[results[(f,n,i,o,b,m)] for i in 1:40]
summary=Any[]; paired=Any[]
for (f,o,n,b) in groups
    sa=group(f,o,n,b,"SA")
    for m in METHODS
        g=group(f,o,n,b,m);final=[r["c_min"] for r in g];calls=[r["objective_calls"] for r in g]
        seconds=[r["elapsed_seconds"] for r in g];gains=[gain(r,r["c_min"]) for r in g]
        push!(summary,(f,o,n,b,m,mean(final),std(final),mean(gains),median(gains),std(gains),mean(calls),std(calls),minimum(calls),maximum(calls),mean(seconds),maximum(seconds),mean(r["completed_iterations"] for r in g),count(r->r["completed_iterations"]==0,g),mean(r["initialization_seconds"] for r in g),mean(last(r["history"])[1] for r in g),count(t->t>b*1.05,seconds),mean(r["allocated_bytes"] for r in g)))
        m=="SA" && continue
        # Remove only numerical ties before rank tests; preserve raw effects.
        delta=[s["c_min"]-r["c_min"] for (s,r) in zip(sa,g)]
        tolerance=[1e-9+1e-11*max(abs(s["c_min"]),abs(r["c_min"])) for (s,r) in zip(sa,g)]
        d=[abs(v)<=t ? 0. : v for (v,t) in zip(delta,tolerance)]
        rel=[100*v/abs(s["c_min"]) for (s,v) in zip(sa,delta)]
        p=all(iszero,d) ? 1. : pvalue(SignedRankTest(d);tail=:right)
        p2=all(iszero,d) ? 1. : pvalue(SignedRankTest(d);tail=:both)
        push!(paired,Dict("family"=>f,"objective"=>o,"n"=>n,"budget"=>b,"method"=>m,"mean_pct"=>mean(rel),"median_pct"=>median(rel),"sd_pct"=>std(rel),"wins"=>count(>(0),d),"ties"=>count(iszero,d),"losses"=>count(<(0),d),"p_raw"=>p,"p_two_sided"=>p2,"calls_ratio"=>mean(s["objective_calls"] for s in sa)/mean(calls)))
    end
end
function holm!(rows,key,outkey)
    indices=sortperm([r[key] for r in rows]);prev=0.
    for (j,k) in enumerate(indices)
        prev=max(prev,min(1.,(length(rows)-j+1)*rows[k][key]));rows[k][outkey]=prev
    end
end
holm!(paired,"p_raw","p_holm72");holm!(paired,"p_two_sided","p_two_holm72")
open(joinpath(OUT,"group_summary.csv"),"w") do io
    println(io,"family,objective,n,budget,method,mean_final,sd_final,mean_gain_pct,median_gain_pct,sd_gain_pct,mean_calls,sd_calls,min_calls,max_calls,mean_seconds,max_seconds,mean_iterations,zero_iterations,mean_initialization_s,mean_last_improvement_s,overrun_5pct,mean_allocated_bytes")
    foreach(r->println(io,join(r,',')),summary)
end
headers=["family","objective","n","budget","method","mean_pct","median_pct","sd_pct","wins","ties","losses","p_raw","p_holm72","p_two_sided","p_two_holm72","calls_ratio"]
open(joinpath(OUT,"paired_comparisons.csv"),"w") do io
    println(io,join(headers,','));foreach(r->println(io,join([r[h] for h in headers],',')),paired)
end
println("Audit and statistics complete. Plotting 36 groups and 12 budget summaries.");flush(stdout)
default(fontfamily="sans-serif",titlefontsize=10,guidefontsize=9,tickfontsize=8,legendfontsize=8)
fmt(x)=@sprintf("%.1f",x)
curveio=open(joinpath(OUT,"aggregate_curves.csv"),"w")
println(curveio,"family,objective,n,budget,method,axis,x,mean_gain_pct,sd_gain_pct")
for (f,o,n,b) in groups
    title="$(f=="assignment" ? "Assignment" : "Set Partitioning") | $o | n=$n | $(b)s | 40 tasks"
    ptime=plot(title="Mean learning curve (band: task SD)",xlabel="Optimization time (s)",ylabel="Gain over initial best (%)",xlims=(0,b),yformatter=fmt,legend=:bottomright)
    pcalls=plot(title="Learning per call (40 tasks on each curve)",xlabel="Objective calls (log scale)",ylabel="Gain over initial best (%)",xscale=:log10,yformatter=fmt,legend=false)
    pbar=plot(title="Final calls: mean +/- task SD",ylabel="Objective calls (log scale)",yscale=:log10,xticks=(1:3,LABELS),legend=false)
    pbox=plot(title="Paired improvement over SA",ylabel="Relative improvement (%)",xticks=(1:2,LABELS[2:3]),yformatter=fmt,legend=:bottomleft)
    for (j,m) in enumerate(METHODS)
        g=group(f,o,n,b,m)
        times=sort(unique(vcat([0.,Float64(b)],[first.(r["history"]) for r in g]...)))
        # Exact union-of-event-time empirical step mean; no interpolation.
        vals=[gain(r,valat(r,t)) for t in times,r in g]
        av=vec(mean(vals;dims=2));sd=vec(std(vals;dims=2))
        plot!(ptime,times,av;ribbon=sd,fillalpha=.09,color=COLORS[j],lw=2,label=LABELS[j],seriestype=:steppost)
        for k in eachindex(times);println(curveio,join((f,o,n,b,m,"time",times[k],av[k],sd[k]),','));end
        # Every task within a method must have reached a query budget. Never
        # extrapolate terminal quality into an unobserved evaluation budget.
        commoncall=minimum(r["objective_calls"] for r in g)
        lower=max([first(r["evaluation_history"])[1] for r in g]...)
        if commoncall>=lower
            xs=sort(unique(vcat([Float64(lower),Float64(commoncall)],[[Float64(c) for (c,_) in r["evaluation_history"] if lower<=c<=commoncall] for r in g]...)))
            v=[gain(r,valat(r,c;calls=true)) for c in xs,r in g];a=vec(mean(v;dims=2));s=vec(std(v;dims=2))
            plot!(pcalls,xs,a;ribbon=s,fillalpha=.09,lw=2,color=COLORS[j],label=LABELS[j],seriestype=:steppost)
            scatter!(pcalls,[last(xs)],[last(a)];color=COLORS[j],markersize=3,label="")
            for k in eachindex(xs);println(curveio,join((f,o,n,b,m,"calls",xs[k],a[k],s[k]),','));end
        end
        c=[r["objective_calls"] for r in g]
        scatter!(pbar,[j],[mean(c)];yerror=[std(c)],color=COLORS[j],markersize=6,label="")
        scatter!(pbar,fill(j+.09,length(c)),c;color=COLORS[j],alpha=.2,markersize=2,label="")
        if j>1
            sa=group(f,o,n,b,"SA");v=[100*(s["c_min"]-r["c_min"])/abs(s["c_min"]) for (s,r) in zip(sa,g)]
            boxplot!(pbox,fill(j-1,40),v;color=COLORS[j],alpha=.5,label="",whisker_range=1.5,outliers=true)
            scatter!(pbox,[j-1],[mean(v)];color=:black,marker=:diamond,markersize=5,label=j==2 ? "Group average" : "")
            scatter!(pbox,[j-1],[median(v)];color=:purple,marker=:hline,markersize=9,label=j==2 ? "Group median" : "")
        end
    end
    hline!(pbox,[0.];color=:gray,linestyle=:dash,label="")
    fig=plot(ptime,pcalls,pbar,pbox;layout=(2,2),size=(1150,760),plot_title=title,margin=5Plots.mm)
    slug="$(f)_$(o)_n$(n)_$(b)s"
    savefig(fig,joinpath(OUT,slug*".png"))
    println(slug);flush(stdout)
end
close(curveio)
for f in ("assignment","partition"),o in ("quadratic","entropic"),n in (10,12,14)
    p=plot(title="$f | $o | n=$n | mean +/- task SD",xlabel="Independent time budget (s)",ylabel="Final objective calls (log scale)",yscale=:log10,xticks=(1:3,["10","50","200"]),legend=:topleft,size=(800,380),margin=5Plots.mm)
    for (j,m) in enumerate(METHODS)
        means=[mean(r["objective_calls"] for r in group(f,o,n,b,m)) for b in (10,50,200)]
        sds=[std(r["objective_calls"] for r in group(f,o,n,b,m)) for b in (10,50,200)]
        plot!(p,collect(1:3).+(j-2)*.025,means;yerror=sds,color=COLORS[j],marker=:circle,label=LABELS[j],lw=2)
    end
    savefig(p,joinpath(OUT,"$(f)_$(o)_n$(n)_budget_calls.png"))
end
println("Finished: ",OUT)
