# Julia/Plots figures for all refreshed ADD weights; no optimization.
using Plots, StatsPlots

function add_relative_panel(a,b;title="",limits=nothing)
    rel=100 .* (a.-b)./abs.(a)
    grouped=reshape(rel,50,5)
    p=boxplot(repeat(1:5;inner=50),rel;label="",fillcolor=palette(:default)[1],
        fillalpha=.75,outliers=true,markercolor=:black,markersize=2,
        xticks=(1:5,string.((12:16).^2)),xlims=(.5,5.5),
        xlabel="Number of binary variables",ylabel="Relative improvement (%)",
        yformatter=x->@sprintf("%.1f",x),title=title,legend=:topleft)
    hline!(p,[0.];color=:gray,linestyle=:dash,label="")
    # Explicit marker styling also controls the legend symbols.
    plot!(p,1:5,vec(median(grouped;dims=1));marker=:diamond,markersize=2.5,
        color=palette(:default)[3],markercolor=palette(:default)[3],
        markerstrokecolor=palette(:default)[3],markerstrokewidth=.3,label="Group median",linewidth=1.5)
    plot!(p,1:5,vec(mean(grouped;dims=1));marker=:circle,markersize=2.5,
        color=palette(:default)[2],markercolor=palette(:default)[2],
        markerstrokecolor=palette(:default)[2],markerstrokewidth=.3,label="Group average",linewidth=1.5)
    isnothing(limits) || ylims!(p,limits)
    p
end

"""Article panel order: GDF, Mixed, Weighted Rank; linear left, quadratic right."""
function plot_add_article(report)
    chosen=Dict()
    for slug in ("gdf","mixed","weighted_rank"), objective in ("linear","quadratic")
        rows=filter(s->s.objective==objective && s.strategy==slug,report.summaries)
        chosen[(slug,objective)]=rows[argmax([s.mean_relative for s in rows])].weight
    end
    values=Float64[]
    for ((slug,objective),w) in chosen
        d=report.results[objective]
        append!(values,100 .* (d.baseline.-d.costs[(slug,w)])./abs.(d.baseline))
    end
    lo,hi=extrema(vcat(values,[0.]));pad=max(1.,.06*(hi-lo));limits=(lo-pad,hi+pad)
    panels=[]
    for slug in ("gdf","mixed","weighted_rank"), objective in ("linear","quadratic")
        d=report.results[objective];w=chosen[(slug,objective)]
        weight_name=Dict("gdf"=>"filter weight", "mixed"=>"mixing weight", "weighted_rank"=>"rank weight")[slug]
        p=add_relative_panel(d.baseline,d.costs[(slug,w)];
            title="$(titlecase(objective)): $(ADD_LABELS[slug]), $weight_name = $w",limits)
        plot!(p;legendfontsize=8,tickfontsize=9,guidefontsize=10,titlefontsize=12)
        push!(panels,p)
    end
    p=plot(panels...;layout=(3,2),size=(1100,1050),margin=4Plots.mm)
    savefig(p,joinpath(report.output,"ADD_selected_strategies.pdf"))
    savefig(p,joinpath(report.output,"ADD_selected_strategies.png"))
    p
end

function plot_add_weights(report,objective,slug)
    d=report.results[objective]
    allrel=vcat([100 .* (d.baseline.-v)./abs.(d.baseline) for v in values(d.costs)]...)
    lo,hi=extrema(vcat(allrel,[0.]));pad=max(1.,.06*(hi-lo));limits=(lo-pad,hi+pad)
    panels=[add_relative_panel(d.baseline,d.costs[(slug,w)];
        title="$(titlecase(objective)) / $(ADD_LABELS[slug]) / weight $w",limits) for w in ADD_GRIDS[slug]]
    p=plot(panels...;layout=(2,2),size=(1100,780),margin=5Plots.mm)
    path=joinpath(report.output,"$(objective)_$(slug)_all_weights")
    savefig(p,path*".pdf");savefig(p,path*".png")
    p
end

function plot_add_selected(report)
    panels=[]
    for objective in ("linear","quadratic")
        rows=sort(filter(s->s.objective==objective,report.summaries);by=s->-s.mean_relative)
        best=first(rows);d=report.results[objective];a=d.baseline;b=d.costs[(best.strategy,best.weight)]
        push!(panels,add_relative_panel(a,b;title="$(titlecase(objective)): $(ADD_LABELS[best.strategy]), weight $(best.weight)"))
        delta=a.-b;grouped=reshape(delta,50,5)
        p=boxplot(repeat(1:5;inner=50),delta;label="",fillcolor=palette(:default)[1],whisker_range=1.0,
            fillalpha=.75,markercolor=:black,markersize=2,xticks=(1:5,string.((12:16).^2)),
            xlims=(.5,5.5),xlabel="Number of binary variables",ylabel="Absolute cost improvement",
            title="$(titlecase(objective)): absolute improvement",legend=:topleft)
        hline!(p,[0.];color=:gray,linestyle=:dash,label="")
        plot!(p,1:5,vec(median(grouped;dims=1));marker=:diamond,color=palette(:default)[3],label="Group median",linewidth=2)
        plot!(p,1:5,vec(mean(grouped;dims=1));marker=:circle,color=palette(:default)[2],label="Group average",linewidth=2)
        push!(panels,p)
    end
    p=plot(panels...;layout=(2,2),size=(1100,780),margin=5Plots.mm)
    savefig(p,joinpath(report.output,"selected_configurations.pdf"))
    savefig(p,joinpath(report.output,"selected_configurations.png"))
    p
end
