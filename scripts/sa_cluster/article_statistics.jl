# Additional, explicitly exploratory paired tests against the penalty baseline.
using JLD2, SparseArrays, Statistics, HypothesisTests
root=normpath(joinpath(@__DIR__,"..",".."))
old=load(joinpath(root,"data","sa_cluster_v2","results","combined.jld2"),"results")
penalty=load(joinpath(root,"data","sa_penalty_v1","results","combined.jld2"),"results")
rows=NamedTuple[]
for f in ("assignment","partition"),o in ("quadratic","entropic"),n in (10,12,14),b in (10,50,200),m in ("U1_data_EELS","U1_full")
    a=[penalty[(f,n,i,o,b,"SA_penalty")]["c_min"] for i in 1:40]
    v=[old[(f,n,i,o,b,m)]["c_min"] for i in 1:40]
    delta=a.-v
    d=[abs(x)<=1e-9+1e-11*max(abs(y),abs(z)) ? 0. : x for (x,y,z) in zip(delta,a,v)]
    p=all(iszero,d) ? 1. : pvalue(SignedRankTest(d);tail=:right)
    push!(rows,(;family=f,objective=o,n,budget=b,method=m,mean_pct=mean(100delta./abs.(a)),
        wins=count(>(0),d),ties=count(iszero,d),losses=count(<(0),d),p_raw=p))
end
adjusted=zeros(length(rows));previous=0.
for (j,k) in enumerate(sortperm(getproperty.(rows,:p_raw)))
    global previous=max(previous,min(1.,(length(rows)-j+1)*rows[k].p_raw));adjusted[k]=previous
end
out=joinpath(root,"examples","sa_cluster_analysis","paired_vs_penalty.csv")
open(out,"w") do io
    println(io,join((keys(first(rows))...,:p_holm72),','))
    for (r,p) in zip(rows,adjusted);println(io,join((values(r)...,p),','));end
end
println("Saved 72 paired comparisons versus penalty SA (one Holm family).")
for m in ("U1_data_EELS","U1_full")
    ids=findall(r->r.method==m && r.budget==200,rows)
    println(m,": 200 s: positive means=",count(i->rows[i].mean_pct>0,ids),"/12; Holm significant=",count(i->adjusted[i]<.05,ids),"/12")
end
