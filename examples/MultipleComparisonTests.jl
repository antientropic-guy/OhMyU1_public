#!/usr/bin/env julia

# Family-wise corrected paired comparisons for the assignment experiments.
# Run from the project root with:
#   julia --project=. examples/MultipleComparisonTests.jl

using HypothesisTests
using JLD2
using OhMyU1                     # needed when JLD2 reconstructs SolverStatistics
using Printf
using Statistics
using StatsBase: tiedrank

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULT_DIR = joinpath(PROJECT_ROOT, "data", "random_assignment")
const RESULT_CACHE = Dict{String,Any}()

"A candidate configuration and the files containing its results."
struct ComparisonSpec
    objective::String
    strategy::String
    w::Float64
    files::Vector{String}
end

"One row in the corrected multiple-comparison report."
struct ComparisonResult
    objective::String
    strategy::String
    w::Float64
    n::Int
    wins::Int
    losses::Int
    ties::Int
    median_improvement::Float64
    hodges_lehmann::Float64
    rank_biserial::Float64
    p_raw::Float64
    p_adjusted::Float64
    reject::Bool
end

result_file(name) = joinpath(RESULT_DIR, name)

function load_result_dictionary(file::String)
    isfile(file) || error("Missing result file: $file")
    return get!(RESULT_CACHE, file) do
        load(file, "res_dict")
    end
end

const BASELINE_FILES = Dict(
    "linear" => result_file("res_best_cost_barrier_false_div_false_r=50.jld2"),
    "quadratic" => result_file("res_best_cost_barrier_false_div_false_r=7_quadratic.jld2"),
)

function default_comparisons()
    linear_best_nonzero = [
        result_file("res_best nonzero mixed_barrier_false_div_false_r=50_w=0.1_0.2_0.4.jld2"),
        result_file("res_best nonzero mixed_barrier_false_div_false_r=50_w=0.3.jld2"),
    ]
    return vcat(
        [ComparisonSpec("linear", "mixed", w,
            [result_file("res_mixed_barrier_false_div_false_r=50_w=0.1_0.2_0.3_0.4.jld2")])
         for w in (0.1, 0.2, 0.3, 0.4)],
        [ComparisonSpec("linear", "best nonzero mixed", w, linear_best_nonzero)
         for w in (0.1, 0.2, 0.3, 0.4)],
        [ComparisonSpec("linear", "weighted rank", w,
            [result_file("res_weighted rank_barrier_false_div_false_r=50_w=0.1_0.35_0.5_0.7.jld2")])
         for w in (0.1, 0.35, 0.5, 0.7)],
        [ComparisonSpec("quadratic", "mixed", w,
            [result_file("res_mixed_barrier_false_div_false_r=7_w=0.1_0.2_0.3_0.4_quadratic.jld2")])
         for w in (0.1, 0.2, 0.3, 0.4)],
        [ComparisonSpec("quadratic", "best nonzero mixed", w,
            [result_file("res_best nonzero mixed_barrier_false_div_false_r=7_w=0.1_0.2_0.3_0.4_quadratic.jld2")])
         for w in (0.1, 0.2, 0.3, 0.4)],
        [ComparisonSpec("quadratic", "weighted rank", w,
            [result_file("res_weighted rank_barrier_false_div_false_r=7_w=0.1_0.35_0.5_0.7_quadratic.jld2")])
         for w in (0.1, 0.35, 0.5, 0.7)],
    )
end

function load_strategy(files::Vector{String}, strategy::String)
    by_weight = Dict{Float64,Any}()
    for file in files
        res_dict = load_result_dictionary(file)
        haskey(res_dict, strategy) || error("Strategy '$strategy' is absent from $file")
        for (w, values) in res_dict[strategy]
            haskey(by_weight, Float64(w)) && error("Duplicate results for strategy='$strategy', w=$w")
            by_weight[Float64(w)] = values
        end
    end
    return by_weight
end

function load_baseline(file::String)
    res_dict = load_result_dictionary(file)
    haskey(res_dict, "best_cost") || error("Strategy 'best_cost' is absent from $file")
    return res_dict["best_cost"]
end

"Collect pairs in a fixed (n, instance) order and fail on incomplete pairing."
function paired_costs(baseline, candidate; n_sizes=12:16, instances=1:50)
    a = Float64[]
    b = Float64[]
    for n in n_sizes, i in instances
        haskey(baseline, n) && haskey(baseline[n], i) ||
            error("Baseline is missing (n=$n, instance=$i)")
        haskey(candidate, n) && haskey(candidate[n], i) ||
            error("Candidate is missing (n=$n, instance=$i)")
        push!(a, Float64(baseline[n][i].c_min))
        push!(b, Float64(candidate[n][i].c_min))
    end
    return a, b
end

"Holm's step-down adjustment; valid under arbitrary dependence."
function holm_adjust(pvalues::AbstractVector{<:Real})
    m = length(pvalues)
    order = sortperm(pvalues)
    adjusted = zeros(Float64, m)
    running_max = 0.0
    for (rank, original_index) in enumerate(order)
        candidate = min(1.0, (m - rank + 1) * Float64(pvalues[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    end
    return adjusted
end

"Hodges-Lehmann pseudomedian: median of all Walsh averages."
function hodges_lehmann(d::AbstractVector{<:Real})
    walsh = Vector{Float64}(undef, length(d) * (length(d) + 1) ÷ 2)
    k = 1
    for i in eachindex(d), j in i:lastindex(d)
        walsh[k] = (d[i] + d[j]) / 2
        k += 1
    end
    return median!(walsh)
end

function rank_biserial(d::AbstractVector{<:Real})
    nonzero = filter(!iszero, d)
    isempty(nonzero) && return 0.0
    ranks = tiedrank(abs.(nonzero))
    wplus = sum(ranks[nonzero .> 0])
    wminus = sum(ranks[nonzero .< 0])
    return (wplus - wminus) / sum(ranks)
end

function raw_comparison(spec::ComparisonSpec; n_sizes=12:16, instances=1:50)
    baseline = load_baseline(BASELINE_FILES[spec.objective])
    candidates = load_strategy(spec.files, spec.strategy)
    haskey(candidates, spec.w) ||
        error("No results for objective=$(spec.objective), strategy=$(spec.strategy), w=$(spec.w)")
    a, b = paired_costs(baseline, candidates[spec.w]; n_sizes, instances)
    d = a .- b  # positive means the candidate has lower (better) cost
    test = SignedRankTest(a, b)
    return (spec=spec, n=length(d), wins=count(>(0), d), losses=count(<(0), d),
            ties=count(iszero, d), median_improvement=median(d),
            hodges_lehmann=hodges_lehmann(d), rank_biserial=rank_biserial(d),
            p_raw=pvalue(test; tail=:right))
end

"""
    run_multiple_comparisons(; family=:global, alpha=0.05, ...)

Run one-sided paired Wilcoxon signed-rank tests for all requested configurations.
`family=:global` applies one Holm correction across linear and quadratic results.
`family=:objective` instead defines one family per objective type; use that only
when linear and quadratic claims were declared as separate confirmatory families.
"""
function run_multiple_comparisons(; specs=default_comparisons(), family=:global,
                                  alpha=0.05, n_sizes=12:16, instances=1:50)
    family in (:global, :objective) || error("family must be :global or :objective")
    raw = [raw_comparison(spec; n_sizes, instances) for spec in specs]
    adjusted = zeros(Float64, length(raw))
    groups = family == :global ? [collect(eachindex(raw))] :
        [findall(x -> x.spec.objective == objective, raw)
         for objective in unique(x.spec.objective for x in raw)]
    for indices in groups
        adjusted[indices] = holm_adjust([raw[i].p_raw for i in indices])
    end
    return [ComparisonResult(x.spec.objective, x.spec.strategy, x.spec.w, x.n,
                             x.wins, x.losses, x.ties, x.median_improvement,
                             x.hodges_lehmann, x.rank_biserial, x.p_raw,
                             adjusted[i], adjusted[i] <= alpha)
            for (i, x) in enumerate(raw)]
end

function print_report(results; io=stdout, alpha=0.05)
    println(io, "One-sided paired Wilcoxon tests: H1 = candidate cost < best_cost")
    println(io, "Holm-adjusted decisions at family-wise alpha = $alpha")
    @printf(io, "%-10s  %-19s  %5s  %4s  %11s  %11s  %8s  %10s  %10s  %s\n",
            "objective", "strategy", "w", "N", "wins/losses", "HL improve",
            "rank-rb", "raw p", "Holm p", "reject")
    for r in results
        wl = "$(r.wins)/$(r.losses)"
        @printf(io, "%-10s  %-19s  %5.2f  %4d  %11s  %11.4g  %8.3f  %10.3g  %10.3g  %s\n",
                r.objective, r.strategy, r.w, r.n, wl, r.hodges_lehmann,
                r.rank_biserial, r.p_raw, r.p_adjusted, r.reject ? "yes" : "no")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    family = "--by-objective" in ARGS ? :objective : :global
    results = run_multiple_comparisons(; family)
    print_report(results)
end
