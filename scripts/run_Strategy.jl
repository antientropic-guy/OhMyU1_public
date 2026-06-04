# ============================================================================
# File: run_Strategy.jl
# Part of: OhMyU1
#
# Description:
#   Experiments runner for the specified strategy. It runs the solver for 50 random instances of the assignment problem for n=12, 13, 14, 15, 16 and saves the results in a dictionary.
#
# Author: Sergei
# Created: May 2026
# ============================================================================


using Pkg
Pkg.activate("")

using Random, Plots, StatsBase, JLD2, OMEinsum, SparseArrays, LinearAlgebra
using OhMyU1
using OhMyU1: ChargeIndex, U1MPS
using OhMyU1: sample_feasible_lp, sample_feasible_lp!, compute_link_charges, compute_site_charges, compute_indices, init_u1_mps, 
orthogonalize!, normalize!, TrainParams, boltzman_probability, train_nondeg!, sample_nondeg_parallel!, cost, u1_norm
using OhMyU1: diversify!, update_index!, graph_dist, random_bilinear_form, random_assignment_vector, fill_assignment_vectors_parallel!,
pick_best_samples, assignment_matrix, generate_random_matrix_vector, SolverParams, graph_dist_global, solve, get_mps, build_mps_from_feasible_samples,
OptimizationProblem, assert_feasible_lp, SolverStatistics, fill_assignment_vectors!
using Statistics, LaTeXStrings, HypothesisTests
using Base.Threads: @spawn

"""
    run_parallel_experiments()

Run parallel experiments for a configurable solver strategy across multiple problem sizes and weight parameters.

# Algorithm
1. Creates a nested dictionary to store results indexed by strategy → weight → n → problem index
2. Iterates over weight parameters w ∈ [0.1, 0.2, 0.3, 0.4] and problem sizes n = 12, 13, 14, 15, 16
3. For each (w, n) combination, runs 50 random instances of the assignment problem
4. Uses bilinear forms loaded from `data/random_bilinear_forms/q_\$(n)_\$(i)_r=\$(r).jld2`
5. Solves each problem using the specified strategy with the solver parameters
6. Saves results and parameters to JLD2 files after each iteration

# Arguments
- `n_sizes::Vector{Int}`: Problem sizes to test (default: [12, 13, 14, 15, 16])
- `r::Int`: Parameter for bilinear forms (default: 7) - they are taken from uniform distribution over [-r, r]
- `strategy::String`: Solver strategy (default: "mixed")
  - "mixed": Mixed Selection combining best cost and graph distance
  - "weighted rank": Weighted rank selection
  - "best nonzero mixed": Best cost with graph distance filter
- `diversify::Bool`: Whether to use EELS (default: false)
- `barrier::Bool`: Whether to filter all samples with 0 graph distance (default: false)
- `mixed_proportion_vec::Vector{Float64}`: Weight parameters for strategy (default: [0.1, 0.2, 0.3, 0.4])

# Returns
- `nothing`: Results are saved to disk in JLD2 format

# Output Files
- `data/random_assignment/res_\$(strategy)_barrier_\$(barrier)_div_\$(diversify)_r=\$(r)_quadratic.jld2`
- `data/random_assignment/params_\$(strategy)_barrier_\$(barrier)_div_\$(diversify)_r=\$(r)_quadratic.jld2`

# Notes
- Uses `@spawn` for parallel execution via Base.Threads
- Results are stored in a thread-safe manner using a ReentrantLock
- Each problem uses quadratic cost function x' * q * x
- The nested loop structure: w (weight) → n (size) → i (problem instance)
"""
function run_parallel_experiments()
    res_dict = Dict{String, # Srategy
    Dict{Float64, # w
    Dict{Int, # n
    Dict{Int, # i (problem number)
    SolverStatistics}}}}()

    n_sizes = [12, 13, 14, 15, 16]
    r = 7
    strategy = "mixed" # available strategies: "mixed" (Mixed Selection), "weighted rank", "best nonzero mixed" (Best Cost + Graph Distance Filter)
    diversify = false
    barrier = false

    mixed_proportion_vec = [0.1, 0.2, 0.3, 0.4]

    # Fill res dict with dump values:
    res_dict[strategy] = Dict{Float64, Dict{Int, Dict{Int, SolverStatistics}}}()
    for w in mixed_proportion_vec
        res_dict[strategy][w] = Dict{Int, Dict{Int, SolverStatistics}}()
        for n in n_sizes
            res_dict[strategy][w][n] = Dict{Int, SolverStatistics}()
        end
    end

    res_lock = ReentrantLock()  # Lock for writing in dictionary;

    # Main parallel cycle body:
    @sync for w in mixed_proportion_vec
        for n in n_sizes
            A = assignment_matrix(n)
            b = ones(Int, 2 * n)
            for i in 1:50
                @spawn begin
                    try
                        solver_params = SolverParams(NUM_GLOBAL_ITER=20, KEEP_NUM_WORST=0.0, LEARNING_RATE=0.05)
                        q = lock(res_lock) do 
                            load("../data/random_bilinear_forms/q_$(n)_$(i)_r=$(r).jld2")["q"]
                        end
                        local_sampler! = (X::AbstractMatrix{Int}) -> fill_assignment_vectors!(X, n)
                        opt_problem = OhMyU1.OptimizationProblem(A=A, b=b, cost_function=x -> x' * q * x, name="train_$(n)_$(i)")
                        res = solve(opt_problem, local_sampler!, solver_params, strategy, diversify=diversify, barrier=barrier,
                        parallel=false, print_stats=false)
                        lock(res_lock) do
                            res_dict[strategy][w][n][i] = res
                            @save "../data/random_assignment/res_$(strategy)_barrier_$(barrier)_div_$(diversify)_r=$(r)_quadratic.jld2" res_dict
                            @save "../data/random_assignment/params_$(strategy)_barrier_$(barrier)_div_$(diversify)_r=$(r)_quadratic.jld2" solver_params
                            @info "Finished: n=$n, i=$i"
                        end
                    catch e
                        @error "Error in task n=$n, i=$i" exception=(e, catch_backtrace())
                    end
                end
            end
        end
    end
    return nothing
end

run_parallel_experiments()