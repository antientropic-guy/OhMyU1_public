#!/usr/bin/env julia
using OhMyU1, Random
using OhMyU1: OptimizationProblem, assignment_matrix, fill_assignment_vectors!, build_mps_from_feasible_samples
root = normpath(joinpath(@__DIR__, "..", ".."))
out = joinpath(root, "data", "protes_comparison", "scaling")
mkpath(out)
open(joinpath(out, "u1_rank_audit.csv"), "w") do io
    println(io, "n,seed,mode,cut,sectors,degeneracy,blocks_total")
    for n in 2:16
        seed = 20261008+n
        Random.seed!(seed)
        initial = zeros(Int,n*n,400)
        fill_assignment_vectors!(initial,n)
        problem = OptimizationProblem(A=assignment_matrix(n),b=ones(Int,2n),cost_function=x->0.0)
        for (mode, enhance) in (("sample", false),("EELS", true))
            mps = build_mps_from_feasible_samples(problem,initial,1,enhance)
            blocks = sum(length(core.Blocks) for core in mps.Cores)
            @assert all(size(block)==(1,1,1) for core in mps.Cores for block in values(core.Blocks))
            for (k, ind) in enumerate(mps.LinkIndices)
                @assert ind.Card == length(ind.Charges)
                println(io, "$n,$seed,$mode,$(k-1),$(ind.Card),1,$blocks")
            end
        end
        flush(io)
    end
end
