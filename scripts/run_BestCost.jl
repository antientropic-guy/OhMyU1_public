# ============================================================================
# File: run_BestCost.jl
# Part of: OhMyU1
#
# Description:
#   Experiments runner for the "best cost" strategy. It runs the solver for 50 random instances of the assignment problem for n=12, 13, 14, 15, 16 and saves the results in a dictionary.
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

Run parallel experiments for the "best cost" strategy across multiple problem sizes.

# Algorithm
1. Creates a nested dictionary to store results indexed by strategy → n → problem index
2. Iterates over problem sizes n = 12, 13, 14, 15, 16
3. For each n, runs 50 random instances of the assignment problem
4. Uses bilinear forms loaded from `data/random_bilinear_forms/q_\$(n)_\$(i)_r=\$(r).jld2`
5. Solves each problem using the specified strategy with the solver parameters
6. Saves results and parameters to JLD2 files after each iteration

# Parameters
- `n_sizes::Vector{Int}`: Problem sizes to test (default: [12, 13, 14, 15, 16])
- `r::Int`: Rank parameter for bilinear forms (default: 7)
- `strategy::String`: Solver strategy (default: "best_cost")
- `diversify::Bool`: Whether to use EELS for buiding MPS (default: false)
- `barrier::Bool`: Whether to exclude all samples with 0 graph distance (default: false)

# Returns
- `nothing`: Results are saved to disk in JLD2 format


# Output Files
- `data/random_assignment/res_\$(strategy)_barrier_\$(barrier)_div_\$(diversify)_r=\$(r)_quadratic.jld2`
- `data/random_assignment/params_\$(strategy)_barrier_\$(barrier)_div_\$(diversify)_r=\$(r)_quadratic.jld2`

# Notes
- Uses `@spawn` for parallel execution via Base.Threads
- Results are stored in a thread-safe manner using a ReentrantLock
- Each problem uses quadratic cost function x' * q * x
"""
function run_parallel_experiments()

    res_dict = Dict{String, # Srategy
    Dict{Int, # n
    Dict{Int, # i (problem number)
    SolverStatistics}}}()

    n_sizes = [12, 13, 14, 15, 16]
    r = 7
    strategy = "best_cost"
    diversify = false
    barrier = false


    res_dict[strategy] = Dict{Int, Dict{Int, SolverStatistics}}()
    for n in n_sizes
        res_dict[strategy][n] = Dict{Int, SolverStatistics}()
    end
    
    res_lock = ReentrantLock()  # Lock for writing in dictionary;

    @sync for n in n_sizes
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
                        res_dict[strategy][n][i] = res
                        @save "../data/random_assignment/res_$(strategy)_barrier_$(barrier)_div_$(diversify)_r=$(r)_quadratic.jld2" res_dict
                        @save "data/random_assignment/params_$(strategy)_barrier_$(barrier)_div_$(diversify)_r=$(r)_quadratic.jld2" solver_params
                        @info "Finished: n=$n, i=$i"
                    end
                catch e
                    @error "Error in task n=$n, i=$i" exception=(e, catch_backtrace())
                end
            end
        end
    end
    return nothing
end

run_parallel_experiments()