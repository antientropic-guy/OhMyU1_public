#!/usr/bin/env julia
using JLD2, Statistics, Plots, Printf
ENV["GKSwstype"]="100"
root=normpath(joinpath(@__DIR__,".."))
data=joinpath(root,"data","set_partitioning","locality_grid_v1")
out=joinpath(root,"examples","partition_locality_grid");mkpath(out)
rows=load(joinpath(data,"manifest.jld2"),"rows")
@assert length(rows)==945
function summarize(g;w,m=0,n=0)
    a=getproperty.(g,:maxrank);b=getproperty.(g,:meanrank)
    (;w,m,n,count=length(g),mean_max=mean(a),sd_max=std(a),mean_mean=mean(b),sd_mean=std(b))
end
groups=[summarize(filter(r->r.w==w&&r.m==m&&r.n==n,rows);w,m,n) for w in (5,6,8,10,12,15,18) for m in 18:20 for n in 13:15]
pooled=[summarize(filter(r->r.w==w,rows);w) for w in (5,6,8,10,12,15,18)]
for (name,table) in (("rank_groups",groups),("rank_pooled",pooled))
    open(joinpath(out,name*".csv"),"w") do io
        println(io,join(keys(first(table)),","));foreach(r->println(io,join(values(r),",")),table)
    end
end
panels=[]
for (mu,sd,title) in ((:mean_max,:sd_max,"Maximum internal rank"),(:mean_mean,:sd_mean,"Mean internal rank"))
    push!(panels,plot(getproperty.(pooled,:w),getproperty.(pooled,mu);
        yerror=getproperty.(pooled,sd),marker=:circle,lw=2,color=mu==:mean_max ? 1 : 2,
        xlabel="Interaction span w",ylabel="Exact indicator rank (log scale)",yscale=:log10,title,
        label="Mean ± sample SD (135 instances)",legend=:topleft,xticks=[5,6,8,10,12,15,18]))
end
p=plot(panels...;layout=(1,2),size=(1100,440),bottom_margin=5Plots.mm,left_margin=5Plots.mm)
savefig(p,joinpath(out,"rank_vs_span.pdf"));savefig(p,joinpath(out,"rank_vs_span.png"))
println("Saved all 63 groups and seven pooled summaries to $out")
