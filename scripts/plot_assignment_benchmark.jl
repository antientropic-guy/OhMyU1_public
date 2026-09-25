#!/usr/bin/env julia

using Plots
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function read_simple_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty benchmark file: $path")
    header = split(first(lines), ',')
    return [Dict(header .=> split(line, ',')) for line in Iterators.drop(lines, 1) if !isempty(line)]
end

function main(args)
    input = isempty(args) ? joinpath(REPO_ROOT, "examples", "assignment_benchmark.csv") : abspath(args[1])
    output = length(args) < 2 ?
        joinpath(REPO_ROOT, "examples", "pictures", "LibsComparison_fair.pdf") : abspath(args[2])

    rows = read_simple_csv(input)
    ohmy = sort(filter(row -> row["backend"] == "OhMyU1", rows); by=row -> parse(Int, row["n"]))
    constraintnet = sort(filter(row -> row["backend"] == "ConstrainTNet", rows); by=row -> parse(Int, row["n"]))
    length(ohmy) == length(constraintnet) || error("backend result counts differ")

    ns = parse.(Int, getindex.(ohmy, "n"))
    ns == parse.(Int, getindex.(constraintnet, "n")) || error("backend n grids differ")
    t_arr = parse.(Float64, getindex.(constraintnet, "balanced_serial_s"))
    t_ref = parse.(Float64, getindex.(ohmy, "balanced_serial_s"))

    figure = plot(ns .^ 2, t_arr,
        marker=(:diamond, 8),
        color=palette(:default)[7],
        yscale=:log10,
        label="ConstrainTNet",
        title="# Constraints",
        titlefont=11,
    )

    y_ticks = Any[0.1, 1, 10, 100]
    plot!(figure, ns .^ 2, t_ref,
        marker=(:circle, 4),
        color=palette(:default)[3],
        label="OhMyU1",
        yticks=(y_ticks, string.(y_ticks)),
        xlabel="# Variables",
        ylabel="Wall time, s",
        ylims=(0.005, 400),
        xlims=(9, 110),
    )

    y1 = 400
    tick_positions2 = ns .^ 2
    tick_labels2 = string.(2 .* ns)
    for (xi, label) in zip(tick_positions2, tick_labels2)
        plot!(figure, [xi, xi], [300, 400], color=:black, label=false, lw=1)
        annotate!(figure, xi, 200, text(label, 8))
    end

    plot!(figure, [9, 110], [y1, y1], color=:black, lw=1, label=false)
    plot!(figure, [110, 110], [0.005, 400], color=:black, lw=1, label=false)

    mkpath(dirname(output))
    savefig(figure, output)
    @printf("Saved main figure to %s\n", output)
end

main(ARGS)
