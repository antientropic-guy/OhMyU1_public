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
    solve_different_assignment(n_vector, res_dict; time_limit=5.0, T_size=500, num_random_problems=10)

Solve assignment problems of varying sizes and measure the effect of diversification on MPS construction.

# Arguments
- `n_vector::Vector{Int64}`: Vector of problem sizes (n) to test. Density is computed as `d = 1/n`.
- `res_dict::Dict{Float64, Dict{String, Vector{Float64}}}`: Results dictionary to populate
- `time_limit::Float64`: Time limit in seconds for feasibility check (default: 5.0)
- `T_size::Int`: Number of feasible samples to generate for MPS construction (default: 500)
- `num_random_problems::Int`: Number of random problems to solve per size (default: 10)

# Returns
- `res_dict::Dict{Float64, Dict{String, Vector{Float64}}}`: Updated results dictionary with:
  - `with_diff_num`: Squared norm after diversification (applying EELS)
  - `no_diff_num`: Squared norm before diversification (only Method (ii))
  - `with_diff_time`: Time for TT construction + diversification
  - `no_diff_time`: Time for TT construction only

# Notes
- Uses assignment matrix structure where each row/column must have exactly one assignment
- The density decreases as n increases (more sparse problems for larger n)
- Results are saved incrementally to allow recovery from interruptions
"""
function solve_different_assignment(n_vector::Vector{Int64}, res_dict::Dict{Float64, Dict{String, Vector{Float64}}};
    time_limit::Float64=5.0, T_size::Int=500, num_random_problems::Int=10)

    for n in n_vector
        d = 1 / n
        res_dict[d] = Dict("with_diff_num" => Float64[], "no_diff_num" => Float64[], 
        "with_diff_time" => Float64[], "no_diff_time" => Float64[])
    end

    for n in n_vector
        d = 1 / n
        println("Solving for n = $n (density = $d)")
        A = assignment_matrix(n)
        b = ones(Int, 2 * n)
        optimization_problem = OptimizationProblem(A=A, b=b, cost_function=x->1.0, name="Assignment problem with n = $n")

        for _ in 1:num_random_problems
            T = zeros(Int, n ^ 2, T_size)  ### Memory allocation;
            fill_assignment_vectors_parallel!(T, n)
            t1 = @elapsed begin
            mps = build_mps_from_feasible_samples(optimization_problem, T, 1, false)
            orthogonalize!(mps)
            end
            push!(res_dict[d]["no_diff_time"], t1)
            push!(res_dict[d]["no_diff_num"], u1_norm(mps) ^ 2)
            println("Before diversification: ", u1_norm(mps) ^ 2)
            t2 = @elapsed begin
            diversify!(mps)
            end
            mps.ort_center = 0
            orthogonalize!(mps)
            push!(res_dict[d]["with_diff_time"], t2 + t1)
            push!(res_dict[d]["with_diff_num"], u1_norm(mps) ^ 2)
            println("After diversification: ", u1_norm(mps) ^ 2, "\n")

            @save "../data/assignment_results/res_sparse_assign_large_points_T_$(T_size).jld2" res_dict

        end
    end
    return res_dict
end

res_dict1 = load("../data/assignment_results/res_sparse_assign_T_10000.jld2", "res_dict")
solve_different_assignment([25, 30], res_dict1, T_size=10000)

res_dict2 = load("../data/assignment_results/res_sparse_assign_T_4000.jld2", "res_dict")
solve_different_assignment([30], res_dict2, T_size=4000)

