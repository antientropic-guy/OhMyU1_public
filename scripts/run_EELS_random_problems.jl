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
    generate_random_matrix_vector_sparse(m, N, r; density=0.1)

Generate a sparse random constraint matrix `A` and constraint vector `b` for a linear program.

# Arguments
- `m::Int`: Number of constraints (rows)
- `N::Int`: Number of variables (columns)
- `r::Int`: Range of non-zero values; values are sampled from `-r:-1` and `1:r`
- `density::Float64`: Fraction of elements that are non-zero (default: 0.1)

# Returns
- `A::Matrix{Int}`: Sparse constraint matrix of size (m, N)
- `b::Vector{Int}`: Sparse constraint vector of length m

# Example
```julia
A, b = generate_random_matrix_vector_sparse(10, 75, 2; density=0.2)
```
"""
function generate_random_matrix_vector_sparse(m, N, r; density=0.1)
    maskA = rand(m, N) .< density
    maskb = rand(m)    .< density
    vals = vcat(-r:-1, 1:r)

    A = zeros(Int, m, N)
    b = zeros(Int, m)

    A[maskA] .= rand(vals, sum(maskA))
    b[maskb] .= rand(vals, sum(maskb))
    return A, b
end

"""
    solve_with_different_sparsity(density, res_dict; N=75, m=10, r=1, time_limit=8.0, num_random_problems=10, T_size=500)

Solve linear programs with varying sparsity levels and measure diversification effects.

# Arguments
- `density::Vector{Float64}`: Vector of sparsity density values to test
- `res_dict::Dict{Float64, Dict{String, Vector{Float64}}}`: Results dictionary to populate
- `N::Int`: Number of variables (columns) in the constraint matrix (default: 75)
- `m::Int`: Number of constraints (rows) in the constraint matrix (default: 10)
- `r::Int`: Range of non-zero values; values are sampled from `-r:-1` and `1:r` (default: 1)
- `time_limit::Float64`: Time limit in seconds for feasibility check and sampling (default: 8.0)
- `num_random_problems::Int`: Number of random problems to generate per density (default: 10)
- `T_size::Int`: Number of feasible samples to generate for MPS construction (default: 500)

# Returns
- `res_dict::Dict{Float64, Dict{String, Vector{Float64}}}`: Updated results dictionary with:
  - `with_diff_num`: Squared norm after applying EELS
  - `no_diff_num`: Squared norm before applying EELS (only Method (ii))
  - `with_diff_time`: Time for TT construction + EELS
  - `no_diff_time`: Time for TT construction only (Method (ii))

# Notes
- Skips infeasible LP problems (verified via `assert_feasible_lp`)
- Requires at least 95% of requested samples to be feasible
- Saves each problem to `data/random_problems_n_\$(N)_m_\$(m)_r_\$(r)/problem_density_\$(d)_\$(i).jld2`
"""
function solve_with_different_sparsity(density::Vector{Float64}, res_dict::Dict{Float64, Dict{String, Vector{Float64}}}; 
    N::Int=75, m::Int=10, r::Int=1, time_limit::Float64=8.0, num_random_problems::Int=10, T_size::Int=500)

    for d in density
        res_dict[d] = Dict("with_diff_num" => Float64[], "no_diff_num" => Float64[],
        "with_diff_time" => Float64[], "no_diff_time" => Float64[])
    end


    for d in density
        i = 1
        println("\n=== Density ", d, " ===")
        while i <= num_random_problems
            A, b = generate_random_matrix_vector_sparse(m, N, r; density=d)

            if !assert_feasible_lp(A, b, time_limit)  # time_limit seconds time limit for feasibility check
                println("Generated infeasible LP, skipping...")
                continue
            end

            @save "../data/random_problems_n_$(N)_m_$(m)_r_$(r)/problem_density_$(d)_$(i).jld2" A b
            i += 1


            optimization_problem = OptimizationProblem(A=A, b=b, cost_function=x->1.0, name="Random sparse LP $(i)")
            T = sample_feasible_lp(A, b, T_size, time_limit=time_limit)
            @assert size(T, 2) > 0.95 * T_size "Not enough feasible samples generated for density $d, problem $i. Got $(size(T, 2)) samples."

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

            @save "../data/random_problems_n_$(N)_m_$(m)_r_$(r)/res_sparse.jld2" res_dict

        end
    end
    return res_dict
end

d_vec = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
res_dict = Dict{Float64, Dict{String, Vector{Float64}}}()
N = 75
r = 2
T_size = 400

solve_with_different_sparsity(d_vec, res_dict, N=N, m=10, r=r, time_limit=9.0, T_size=T_size)
