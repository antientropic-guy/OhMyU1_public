# Paired analysis of the refreshed ADD training sweep. No optimization is run.
using OhMyU1, JLD2, SHA, Statistics, LinearAlgebra, HypothesisTests, Printf, Markdown

const ADD_ROOT=normpath(joinpath(@__DIR__,".."))
const ADD_LABELS=Dict("mixed"=>"Mixed","weighted_rank"=>"Weighted Rank","gdf"=>"GDF")
const ADD_NAMES=Dict("mixed"=>"mixed","weighted_rank"=>"weighted rank","gdf"=>"best nonzero mixed")
const ADD_GRIDS=Dict("mixed"=>[.1,.2,.3,.4],"weighted_rank"=>[.1,.35,.5,.7],"gdf"=>[.1,.2,.3,.4])

function holm_adjust(p)
    order=sortperm(p);out=zeros(length(p));running=0.
    for (j,i) in enumerate(order)
        running=max(running,min(1.,(length(p)-j+1)*p[i]));out[i]=running
    end
    out
end
function paired_summary(a,b)
    delta=a.-b
    if all(iszero,delta)
        W=0.;p=1.;two=1.
    else
        t=SignedRankTest(a,b)
        W=Float64(t.W);p=pvalue(t;tail=:right);two=pvalue(t;tail=:both)
    end
    @assert all(!iszero,a)
    rel=100 .* delta ./ abs.(a)
    (;N=length(a),wins=count(>(0),delta),ties=count(iszero,delta),losses=count(<(0),delta),
      W,p,two,mean_relative=mean(rel),median_relative=median(rel),mean_absolute=mean(delta))
end
percent(x)=@sprintf("%.1f%%",x)
ptext(x)=x<.001 ? @sprintf("%.2e",x) : @sprintf("%.4f",x)
function write_csv(path,rows)
    isempty(rows) && error("No rows to export")
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows;println(io,join(string.(values(row)),','));end
    end
end

function analyze_add(;input=joinpath(ADD_ROOT,"data","assignment_add_tuning_v1","1f4c7d9256d7"),
                     output=joinpath(ADD_ROOT,"examples","add_tuning_analysis"))
    BLAS.set_num_threads(1);mkpath(output)
    ns=12:16;ids=1:50;indices=[(n,i) for n in ns for i in ids]
    results=Dict{String,Any}();summaries=NamedTuple[];groups=NamedTuple[];pairs=NamedTuple[]
    audit=String[];file_hashes=NamedTuple[];total=0;codehash=nothing;parameters=nothing
    for objective in ("linear","quadratic")
        files=Dict(slug=>load(joinpath(input,"res_$(slug)_$(objective).jld2")) for slug in ("best_cost","mixed","weighted_rank","gdf"))
        base=files["best_cost"];cfg=base["configuration"]
        if isnothing(codehash);codehash=cfg.code_hash;parameters=cfg.parameters;end
        @assert cfg.ns==collect(ns) && cfg.ids==collect(ids) && cfg.split=="train"
        costs=Dict{Tuple{String,Float64},Vector{Float64}}()
        baseline=Float64[base["res_dict"]["best_cost"][n][i].c_min for (n,i) in indices]
        coef=Dict{Tuple{Int,Int},Any}()
        for (n,i) in indices
            p=objective=="linear" ? joinpath(ADD_ROOT,"data","random_vectors","c_$(n)_$(i)_r=50.jld2") :
                joinpath(ADD_ROOT,"data","random_bilinear_forms","q_$(n)_$(i)_r=7.jld2")
            @assert bytes2hex(sha256(read(p)))==cfg.hashes[(n,i)]
            coef[(n,i)]=load(p,objective=="linear" ? "c" : "q")
        end
        for slug in ("best_cost","mixed","weighted_rank","gdf")
            d=files[slug];c=d["configuration"]
            @assert c.code_hash==codehash && c.parameters==parameters && c.hashes==cfg.hashes
            @assert c.objective==objective && c.split=="train" && c.ns==cfg.ns && c.ids==cfg.ids
            @assert c.seed_rule==cfg.seed_rule && c.julia_version==cfg.julia_version && c.blas_threads==1
            weights=slug=="best_cost" ? [0.] : ADD_GRIDS[slug]
            name=slug=="best_cost" ? "best_cost" : ADD_NAMES[slug]
            @assert c.strategy==name && c.weights==weights
            @assert Set(keys(d["res_dict"]))==Set([name])
            @assert Set(keys(d["diagnostics"]))==Set((w,n,i) for w in weights for (n,i) in indices)
            slug!="best_cost" && @assert Set(keys(d["res_dict"][name]))==Set(weights)
            for w in weights
                bucket=slug=="best_cost" ? d["res_dict"][name] : d["res_dict"][name][w]
                @assert Set(keys(bucket))==Set(ns)
                for n in ns;@assert Set(keys(bucket[n]))==Set(ids);end
                values=Float64[]
                for (n,i) in indices
                    r=bucket[n][i];info=d["diagnostics"][(w,n,i)];ref=base["diagnostics"][(0.,n,i)]
                    @assert info.seed==ref.seed && info.initial_training_sha256==ref.initial_training_sha256
                    @assert r.learning_curve[1]==base["res_dict"]["best_cost"][n][i].learning_curve[1]
                    @assert length(r.learning_curve)==parameters.NUM_GLOBAL_ITER+1
                    @assert all(diff(r.learning_curve).<=1e-9) && last(r.learning_curve)==r.c_min
                    x=r.incub;@assert length(x)==n*n && all(v->v in (0,1),x)
                    @assert all(OhMyU1.assignment_matrix(n)*x .== 1)
                    f=coef[(n,i)];value=objective=="linear" ? dot(f,x) : dot(x,f*x)
                    @assert isfinite(r.c_min) && isapprox(value,r.c_min;atol=1e-8,rtol=1e-12)
                    push!(values,r.c_min);total+=1
                end
                if slug!="best_cost"
                    costs[(slug,w)]=values;s=paired_summary(baseline,values)
                    push!(summaries,merge((objective=objective,strategy=slug,weight=w),s))
                    for (k,n) in enumerate(ns)
                        r=(50(k-1)+1):50k
                        push!(groups,merge((objective=objective,strategy=slug,weight=w,n=n),paired_summary(baseline[r],values[r])))
                    end
                end
            end
            path=joinpath(input,"res_$(slug)_$(objective).jld2")
            push!(file_hashes,(file=basename(path),sha256=bytes2hex(sha256(read(path)))))
        end
        configs=sort(collect(keys(costs)))
        for j in 1:length(configs), k in j+1:length(configs)
            a,b=configs[j],configs[k]
            push!(pairs,merge((objective=objective,a_strategy=a[1],a_weight=a[2],b_strategy=b[1],b_weight=b[2]),paired_summary(costs[a],costs[b])))
        end
        results[objective]=(;baseline,costs)
        push!(audit,"- **$(titlecase(objective))**: 250 Best Cost and 3,000 ADD outcomes; all expected keys present; objective/input hashes, initial-data hashes, seeds, initial costs, feasibility, final costs and 20 completed iterations verified.")
    end
    adjglobal=holm_adjust([s.p for s in summaries])
    adjusted=NamedTuple[]
    for objective in ("linear","quadratic")
        ii=findall(s->s.objective==objective,summaries);adj=holm_adjust([summaries[i].p for i in ii])
        for (k,i) in enumerate(ii)
            push!(adjusted,merge(summaries[i],(holm12=adj[k],holm24=adjglobal[i])))
        end
    end
    adjusted_pairs=NamedTuple[]
    for objective in ("linear","quadratic")
        rows=filter(s->s.objective==objective,pairs);adj=holm_adjust([s.two for s in rows])
        for (i,s) in enumerate(rows);push!(adjusted_pairs,merge(s,(holm66_two_sided=adj[i],)));end
    end
    write_csv(joinpath(output,"all_configurations.csv"),adjusted)
    write_csv(joinpath(output,"by_size.csv"),groups)
    write_csv(joinpath(output,"pairwise_configurations.csv"),adjusted_pairs)
    write_csv(joinpath(output,"input_hashes.csv"),file_hashes)
    raw=[(objective=o,n=n,instance=i,strategy=s,weight=w,baseline=results[o].baseline[j],candidate=v[j])
        for o in ("linear","quadratic") for ((s,w),v) in sort(collect(results[o].costs);by=first) for (j,(n,i)) in enumerate(indices)]
    write_csv(joinpath(output,"paired_costs.csv"),raw)
    open(joinpath(output,"audit.md"),"w") do io
        println(io,"## Input audit\n\nSource: `$(relpath(input,ADD_ROOT))`. Recorded implementation: `$(codehash)`.\n")
        println(io,join(audit,"\n"))
        println(io,"\nAll **$total** outcomes are complete. The runner explicitly schedules Best Cost as array task 0, once per training instance and objective (500 baseline runs total). There are 6,000 ADD runs. No test data, old result files or optimization reruns enter this report.\n")
        println(io,"Recorded solver parameters: `$(parameters)`. Both families use 1,000 initial feasible points, 10,000 MPS samples, 20 global iterations, one sweep, learning rate 0.05, elite fraction 0.05, and no worst retention. The runner disables EELS and the barrier. ADD and Best Cost both receive external feasible samples; this is the ADD comparison, not the closed-data EELS comparison.")
    end
    for objective in ("linear","quadratic")
        rows=sort(filter(s->s.objective==objective,adjusted);by=s->-s.mean_relative)
        open(joinpath(output,"$(objective)_table.md"),"w") do io
            println(io,"## $(titlecase(objective)): every strategy and weight\n")
            println(io,"Positive improvement means lower cost than Best Cost. W / T / L counts strict wins / exact ties / losses; percentages use all 250 pairs. Rows are ordered by mean relative improvement.\n")
            println(io,"| Strategy | Weight | W / T / L | Win rate | Mean improvement | Median improvement | W statistic | One-sided p | Holm (12) | Holm (24) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for s in rows
                println(io,"| $(ADD_LABELS[s.strategy]) | $(s.weight) | $(s.wins) / $(s.ties) / $(s.losses) | $(percent(100s.wins/s.N)) | $(percent(s.mean_relative)) | $(percent(s.median_relative)) | $(s.W) | $(ptext(s.p)) | $(ptext(s.holm12)) | $(ptext(s.holm24)) |")
            end
        end
        println("$(uppercase(objective)) rankings:")
        for s in rows;println("$(s.strategy) w=$(s.weight): mean=$(percent(s.mean_relative)), median=$(percent(s.median_relative)), W/T/L=$(s.wins)/$(s.ties)/$(s.losses), p=$(s.p), Holm24=$(s.holm24)");end
    end
    report=(;results,summaries=adjusted,groups,pairs=adjusted_pairs,output,parameters)
    write_add_recommendations(report)
    report
end

function write_add_recommendations(report)
    open(joinpath(report.output,"recommendations.md"),"w") do io
        println(io,"## Selection and interpretation\n")
        println(io,"Selection rule: choose the configuration with the largest mean per-instance relative improvement against the common baseline, using the Wilcoxon comparisons to assess evidence. This rule is stated for this analysis, not claimed to have been preregistered. A smaller p-value is not a larger effect.\n")
        println(io,"| Objective | Best weight within strategy | Weight | Mean improvement | Median improvement | W / T / L |\n|---|---|---:|---:|---:|---:|")
        winners=[]
        for objective in ("linear","quadratic")
            rows=filter(s->s.objective==objective,report.summaries)
            push!(winners,rows[argmax([s.mean_relative for s in rows])])
            for slug in ("mixed","weighted_rank","gdf")
                rr=filter(s->s.strategy==slug,rows);s=rr[argmax([r.mean_relative for r in rr])]
                println(io,"| $(titlecase(objective)) | $(ADD_LABELS[slug]) | $(s.weight) | $(percent(s.mean_relative)) | $(percent(s.median_relative)) | $(s.wins) / $(s.ties) / $(s.losses) |")
            end
        end
        println(io,"\nRecommended configurations:\n")
        for s in winners
            println(io,"- **$(titlecase(s.objective)): $(ADD_LABELS[s.strategy]), weight $(s.weight)**. Mean improvement $(percent(s.mean_relative)); median $(percent(s.median_relative)); wins $(percent(100s.wins/s.N)); baseline one-sided p=$(ptext(s.p)), Holm across all 24 tests=$(ptext(s.holm24)).")
        end
        if all(s->(s.strategy,s.weight)==(first(winners).strategy,first(winners).weight),winners)
            s=first(winners)
            println(io,"\nA single shared choice is therefore **$(ADD_LABELS[s.strategy]), weight $(s.weight)** (`$(ADD_NAMES[s.strategy])`). This is the best tested configuration by the stated effect-size criterion for both objectives.")
            s.strategy=="gdf" && println(io,"\nImplementation setting: pass `\"best nonzero mixed\"` as the strategy argument to `solve` and set `mixed_proportion=$(s.weight)`.")
        end
        println(io,"\n### How clearly does the selected configuration beat alternatives?\n")
        println(io,"Additional two-sided paired Wilcoxon tests compare all 66 configuration pairs within each objective, with Holm correction over those 66 tests. These are exploratory comparisons on the same tuning data. In this table, wins mean the selected configuration has lower cost than its competitor.\n")
        println(io,"| Objective | Competitor | Weight | Selected W / T / L | Two-sided p | Holm (66) |\n|---|---|---:|---:|---:|---:|")
        for s in winners
            rows=filter(r->r.objective==s.objective,report.summaries)
            competitors=[]
            for slug in ("mixed","weighted_rank","gdf")
                candidates=filter(r->r.strategy==slug && (r.strategy,r.weight)!=(s.strategy,s.weight),rows)
                isempty(candidates) || push!(competitors,candidates[argmax([r.mean_relative for r in candidates])])
            end
            for other in competitors
                rr=only(filter(r->r.objective==s.objective &&
                    Set([(r.a_strategy,r.a_weight),(r.b_strategy,r.b_weight)])==Set([(s.strategy,s.weight),(other.strategy,other.weight)]),report.pairs))
                selected_is_b=(rr.b_strategy,rr.b_weight)==(s.strategy,s.weight)
                wins=selected_is_b ? rr.wins : rr.losses;losses=selected_is_b ? rr.losses : rr.wins
                println(io,"| $(titlecase(s.objective)) | $(ADD_LABELS[other.strategy]) | $(other.weight) | $wins / $(rr.ties) / $losses | $(ptext(rr.two)) | $(ptext(rr.holm66_two_sided)) |")
            end
        end
        for s in winners
            unresolved=String[]
            for rr in report.pairs
                rr.objective==s.objective || continue
                chosen=(s.strategy,s.weight)
                a=(rr.a_strategy,rr.a_weight);b=(rr.b_strategy,rr.b_weight)
                chosen in (a,b) || continue
                if rr.holm66_two_sided>=.05
                    other=chosen==a ? b : a
                    push!(unresolved,"$(ADD_LABELS[other[1]]) $(other[2]) (adjusted p=$(ptext(rr.holm66_two_sided)))")
                end
            end
            if isempty(unresolved)
                println(io,"\n**$(titlecase(s.objective))**: the selected configuration is separated from every other tested configuration by the Holm-adjusted pairwise tests.")
            else
                println(io,"\n**$(titlecase(s.objective))**: the selected configuration is not statistically separated from $(join(unresolved, "; ")) after correction. Its largest observed mean supports a practical tuning choice, not a claim of uniquely established superiority.")
            end
        end
        println(io,"\n### Selected configuration by problem size\n")
        println(io,"| Objective | n | Variables | Mean improvement | Median improvement | W / T / L |\n|---|---:|---:|---:|---:|---:|")
        for s in winners,r in filter(r->r.objective==s.objective && r.strategy==s.strategy && r.weight==s.weight,report.groups)
            println(io,"| $(titlecase(r.objective)) | $(r.n) | $(r.n^2) | $(percent(r.mean_relative)) | $(percent(r.median_relative)) | $(r.wins) / $(r.ties) / $(r.losses) |")
        end
        significant=count(s->s.holm24<.05,report.summaries)
        println(io,"\n$significant of 24 configurations beat Best Cost at alpha=0.05 after Holm correction over the full baseline comparison family. See the full tables for the nonsignificant cases.\n")
        println(io,"The recommendation is for subsequent held-out testing. The tuning data establish a strong improvement over Best Cost, but do not by themselves establish out-of-sample performance or a uniquely optimal continuous weight. In particular, inspect the adjusted pairwise p-values before claiming separation from nearby weights. There is one solver seed per objective instance, so this study does not separately estimate within-instance seed variability.\n")
        println(io,"The Wilcoxon test matches the old notebooks exactly: paired raw cost differences, one-sided right tail for baseline minus candidate, exact zero differences treated by HypothesisTests. It is not a t-test of mean percentage improvement. Pooling sizes follows the old reports; larger-magnitude differences can influence rank sums, so the separate size summaries are also supplied. Its usual signed-rank interpretation assumes independent problem pairs and a symmetric difference distribution under a location-shift model.\n")
        println(io,"The comparison uses equal iteration/sample settings, not equal wall-clock budgets. Concurrent timing diagnostics are not isolated speed benchmarks. Shared initial-data hashes verify the starting samples, but subsequent external-source batch sizes may differ by strategy; do not interpret the full adaptive trajectories as identical random draws.")
    end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    analyze_add()
end
