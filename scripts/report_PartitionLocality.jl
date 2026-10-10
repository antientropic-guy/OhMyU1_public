#!/usr/bin/env julia
ENV["GKSwstype"]="100"
include("generate_PartitionLocality.jl")
using Plots,Printf
const LOCALITY_REPORT=joinpath(PARTITION_ROOT,"examples","partition_locality")
function report_locality()
    mkpath(LOCALITY_REPORT)
    default(left_margin=10Plots.mm,bottom_margin=8Plots.mm,right_margin=5Plots.mm,
        top_margin=5Plots.mm,legendfontsize=9,guidefontsize=11,titlefontsize=12)
    rows=NamedTuple[];individual=NamedTuple[]
    for span in LOCALITY_SPANS
        data=[load(locality_path(span,i)) for i in 1:40]
        mx=[d["structural"].maxrank for d in data]
        av=[d["structural"].meanrank for d in data]
        cv=[d["initial_coverage"] for d in data]
        counts=[d["structural"].total for d in data]
        push!(rows,(;span,instances=40,mean_max=mean(mx),median_max=median(mx),sd_max=std(mx),
            mean_mean=mean(av),median_mean=median(av),sd_mean=std(av),
            mean_initial_coverage=mean(cv),min_covers=minimum(counts),max_covers=maximum(counts)))
        for (i,d) in enumerate(data)
            push!(individual,(;span,i,maxrank=mx[i],meanrank=av[i],covers=counts[i],
                initial_charge_coverage=cv[i],initial_unique=size(unique(d["initial"];dims=2),2)))
        end
    end
    for (file,records) in (("summary.csv",rows),("instances.csv",individual))
        open(joinpath(LOCALITY_REPORT,file),"w") do io
            println(io,join(string.(keys(first(records))),","))
            foreach(r->println(io,join(values(r),",")),records)
        end
    end
    p=plot(;xlabel="Maximum tile span",ylabel="Exact full-indicator bond dimension",
        yscale=:log10,xticks=LOCALITY_SPANS,size=(940,530),legend=:topleft,
        title="Set partitioning: 225 variables, 18 constraints, 40 instances per span")
    for (color,mn,md,sd,label) in ((1,:mean_max,:median_max,:sd_max,"Maximum across cuts"),
            (2,:mean_mean,:median_mean,:sd_mean,"Mean across cuts"))
        y=getproperty.(rows,mn)
        plot!(p,LOCALITY_SPANS,y;yerror=getproperty.(rows,sd),color,marker=:circle,
            linewidth=2,label="$label: group mean +/- SD")
        plot!(p,LOCALITY_SPANS,getproperty.(rows,md);color,linestyle=:dash,
            marker=:diamond,markersize=3,label="$label: group median")
    end
    # Analytical row-major assignment comparator, 225 variables, 30 constraints.
    bn(n,k)=0<=k<=n ? binomial(n,k) : 0
    ar=[begin r,s=divrem(k,15);bn(15,r)-bn(s,r-(15-s))+bn(15,r+1)-bn(15-s,r+1) end for k in 1:224]
    hline!(p,[maximum(ar)];color=:gray,linestyle=:dot,label="Assignment n=15: maximum rank ($(maximum(ar)))")
    savefig(p,joinpath(LOCALITY_REPORT,"ranks.pdf"));savefig(p,joinpath(LOCALITY_REPORT,"ranks.png"))
    q=plot(LOCALITY_SPANS,100getproperty.(rows,:mean_initial_coverage);
        marker=:circle,linewidth=2,label="Group mean",xlabel="Maximum tile span",
        ylabel="Initial charge coverage",yformatter=x->@sprintf("%.1f%%",x),
        xticks=LOCALITY_SPANS,ylims=(0,105),size=(850,450),legend=:topright,
        title="Coverage of full-indicator charges by 500 uniform initial samples")
    plot!(q,LOCALITY_SPANS,[100median([r.initial_charge_coverage for r in individual if r.span==s]) for s in LOCALITY_SPANS];
        marker=:diamond,linestyle=:dash,label="Group median")
    savefig(q,joinpath(LOCALITY_REPORT,"initial_coverage.pdf"));savefig(q,joinpath(LOCALITY_REPORT,"initial_coverage.png"))
    jldsave(joinpath(LOCALITY_REPORT,"summary.jld2");rows,individual,assignment_ranks=ar)
    foreach(println,rows)
end
if abspath(PROGRAM_FILE)==@__FILE__
    report_locality()
end
