# ============================================================================
# File: tools.jl
# Part of: OhMyU1
#
# Description:
#   Helper-functions for the library. 
#
# Author: Sergei
# Created: Jan 2026
# ============================================================================

function assert_feasible_lp(A::Matrix{T}, b::Vector{T}, time_limit::Float64) where {T<:Integer}
    n = size(A, 2)
    m = size(A, 1)

    model = Model(SCIP.Optimizer)
    set_optimizer_attribute(model, "limits/time", time_limit)   # time limit in seconds
    set_optimizer_attribute(model, "display/verblevel", 0)

    @variable(model, 0 <= x[1:n] <= 1, Int)
    @objective(model, Min, sum(x[j] for j in 1:n))  # dump objective;
    for row in 1:m
        @constraint(model, sum(A[row,j]*x[j] for j in 1:n) == b[row])
    end

    optimize!(model)

    if (termination_status(model) == MOI.OPTIMAL) || (primal_status(model) == MOI.FEASIBLE_POINT)
        return true
    else
        return false
    end
end

"""
    sample_feasible_lp(A::Matrix{T}, b::Vector{T}, count) where {T<:Integer} -> Matrix{T}

Samples random feasible points satisfying Ax = b

# Arguments
- `A::Matrix{T}`: Matrix of the constraints (equalities)
- `b::Vector{T}`: Right hand side of the constraints
- `count`: number of the feasible samples

# Returns
- Matrix{T}`: Feasible points (in columns).

"""
function _configure_scip_reproducibility!(model::Model, rng::AbstractRNG)
    max_scip_seed = UInt32(typemax(Int32))
    next_seed() = Int(rand(rng, UInt32) % max_scip_seed)
    set_optimizer_attribute(model, "randomization/randomseedshift", next_seed())
    set_optimizer_attribute(model, "randomization/permutationseed", next_seed())
    set_optimizer_attribute(model, "randomization/lpseed", next_seed())
    set_optimizer_attribute(model, "parallel/mode", 1)
    set_optimizer_attribute(model, "parallel/maxnthreads", 1)
    return nothing
end

function sample_feasible_lp(rng::AbstractRNG, A::Matrix{T}, b::Vector{T}, count;
                            time_limit::Float64=5.0) where {T<:Integer}
    n = size(A, 2)
    m = size(A, 1)
    results = Matrix{T}(undef, n, count)
    n_filled_cols = 0

    model = Model(SCIP.Optimizer)
    _configure_scip_reproducibility!(model, rng)
    set_optimizer_attribute(model, "limits/time", time_limit)   # 5s timelimit
    set_optimizer_attribute(model, "display/verblevel", 4)
    # set_optimizer_attribute(model, "logfile", "scip.log")

    @variable(model, 0 <= x[1:n] <= 1, Int)
    @objective(model, Min, sum(x[j] for j in 1:n))  # dump objective;
    for row in 1:m
        @constraint(model, sum(A[row,j]*x[j] for j in 1:n) == b[row])
    end

    for i in 1:count
        c = rand(rng, n) .- 0.5
        for j in 1:n
             set_objective_coefficient(model, x[j], c[j])
        end

        open("scip.log", "w") do io
        redirect_stdout(io) do
        optimize!(model)
        end
    end

        # optimize!(model)
        if (termination_status(model) == MOI.OPTIMAL) || (primal_status(model) == MOI.FEASIBLE_POINT)
            results[:, n_filled_cols + 1] = round.(T, value.(x))
            n_filled_cols += 1
        end
    end
    results = results[:, 1:n_filled_cols]
    results = unique(eachcol(results))
    return reduce(hcat, results)
end

sample_feasible_lp(A::Matrix{T}, b::Vector{T}, count; kwargs...) where {T<:Integer} =
    sample_feasible_lp(Random.default_rng(), A, b, count; kwargs...)


"""
    sample_feasible_lp(A::Matrix{T}, b::Vector{T}, results::Matrix{T}, count) where {T<:Integer} -> nothing

Samples random feasible points satisfying Ax = b in pre-allocated memory.

# Arguments
- `A::Matrix{T}`: Matrix of the constraints (equalities)
- `b::Vector{T}`: Right hand side of the constraints
- `results::Matrix{T}`: Allocated memory for storing feasible samples
- `count`: number of the feasible samples

# Returns
- nothing.

"""
function sample_feasible_lp!(rng::AbstractRNG, A::Matrix{T}, b::Vector{T},
                             results::AbstractMatrix{T}, count::Int;
                             time_limit::Float64=5.0) where {T<:Integer}
    n = size(A, 2)
    m = size(A, 1)
    n_filled_cols = 0

    model = Model(SCIP.Optimizer)
    _configure_scip_reproducibility!(model, rng)
    set_optimizer_attribute(model, "limits/time", time_limit)   # time limit in seconds
    set_optimizer_attribute(model, "display/verblevel", 4)
        @variable(model, 0 <= x[1:n] <= 1, Int)
        @objective(model, Min, sum(x[j] for j in 1:n))  # dump objective;
        for row in 1:m
            @constraint(model, sum(A[row,j]*x[j] for j in 1:n) == b[row])
        end

    for i in 1:count
        c = rand(rng, n) .- 0.5
        for j in 1:n
             set_objective_coefficient(model, x[j], c[j])
        end
        
        open("scip.log", "w") do io
        redirect_stdout(io) do
        optimize!(model)
        end
    end

        # optimize!(model)
        if (termination_status(model) == MOI.OPTIMAL) || (primal_status(model) == MOI.FEASIBLE_POINT)
            results[:, n_filled_cols + 1] .= T.(value.(x))
            n_filled_cols += 1
        end
    end
    return nothing
end


sample_feasible_lp!(A::Matrix{T}, b::Vector{T}, results::AbstractMatrix{T}, count::Int;
                    kwargs...) where {T<:Integer} =
    sample_feasible_lp!(Random.default_rng(), A, b, results, count; kwargs...)

# Vector of boltzman probabilities
function boltzman_probability(cost_function::Function, x_vector, temperature::Number, norm::Number, min_cost::Number)
    return exp(-(cost_function(x_vector) - min_cost) / temperature) / norm
end

# Linear cost:
function cost(c)
    return t -> dot(c, t)
end


# Random tools:

const _RNG_BLOCK_SIZE = 256

function random_bilinear_form(rng::AbstractRNG, N::Int, density::Float64, r::Number)
    Q = sprand(rng, N, N, density)
    Q.nzval .= rand(rng, length(Q.nzval)) .* (2r) .- r
    Q = 0.5 * (Q + Q')
    return Q
end

random_bilinear_form(N::Int, density::Float64, r::Number) =
    random_bilinear_form(Random.default_rng(), N, density, r)

function random_assignment_vector(rng::AbstractRNG, n::Int)
    p = randperm(rng, n)
    x = zeros(Int, n^2)
    for i in 1:n
        j = p[i]
        idx = (i - 1) * n + j
        x[idx] = 1
    end
    return x
end

random_assignment_vector(n::Int) = random_assignment_vector(Random.default_rng(), n)

function fill_assignment_vectors_parallel!(rng::AbstractRNG, X::AbstractMatrix{Int}, n::Int)
    M = size(X, 2)
    num_blocks = cld(M, _RNG_BLOCK_SIZE)
    block_seeds = rand(rng, UInt64, num_blocks)
    Threads.@threads for block in 1:num_blocks
        block_rng = Random.Xoshiro(block_seeds[block])
        first_column = (block - 1) * _RNG_BLOCK_SIZE + 1
        last_column = min(block * _RNG_BLOCK_SIZE, M)
        for k in first_column:last_column
            p = randperm(block_rng, n)
            fill!(@view(X[:, k]), 0)
            for i in 1:n
                j = p[i]
                idx = (i - 1) * n + j
                X[idx, k] = 1
            end
        end
    end
end

fill_assignment_vectors_parallel!(X::AbstractMatrix{Int}, n::Int) =
    fill_assignment_vectors_parallel!(Random.default_rng(), X, n)

function fill_assignment_vectors!(rng::AbstractRNG, X::AbstractMatrix{Int}, n::Int)
    M = size(X, 2)
    for k in 1:M
        p = randperm(rng, n)
        fill!(@view(X[:, k]), 0)
        for i in 1:n
            j = p[i]
            idx = (i - 1) * n + j
            X[idx, k] = 1
        end
    end
end

fill_assignment_vectors!(X::AbstractMatrix{Int}, n::Int) =
    fill_assignment_vectors!(Random.default_rng(), X, n)

"""
Takes a matrix of (column) samples and a vector of pre-computed costs for these samples (same order), and returns the k best samples according to the costs.
"""
function pick_best_samples(samples::AbstractMatrix{Int}, k::Int, costs_vector::AbstractVector; rev=false)
    new_k = min(size(samples, 2), k)
    idx = partialsortperm(costs_vector, 1:new_k, rev=rev)
    return copy(view(samples, :, idx))
end

function pick_mixed_samples(samples::AbstractMatrix{Int}, k_best::Int, k_worst::Int, costs_vector::AbstractVector; rev=false)
    n = length(costs_vector)
    k_best = min(n, k_best)
    k_worst = min(n - k_best, k_worst)
    idx_best = partialsortperm(costs_vector, 1:k_best, rev=rev)
    if k_worst > 0
        idx_worst = partialsortperm(costs_vector, 1:k_worst, rev=!rev)
        idx = vcat(idx_best, idx_worst)
    else
        idx = idx_best
    end
    return copy(view(samples, :, idx))
end

function assignment_matrix(n::Int)
    A = zeros(Int, 2 * n, n ^ 2)
    for j in 1:n
        for i in 1:n
            A[j, (i - 1) * n + j] = 1
            A[n + i, (i - 1) * n + j] = 1
        end
    end
    return A
end

function generate_random_matrix_vector(rng::AbstractRNG, m, N, r)
    A = rand(rng, -r:r, m, N)
    b = rand(rng, -r:r, m)
    return A, b
end


generate_random_matrix_vector(m, N, r) =
    generate_random_matrix_vector(Random.default_rng(), m, N, r)

function allcols_equal(A::Matrix{Int}, v::Vector{Int})
    @inbounds for j in 1:size(A, 2)
        @inbounds for i in 1:size(A, 1)
            A[i, j] == v[i] || return false
        end
    end
    return true
end

function allcols_AX_equal_b(A, X, b)
    m, n = size(A)
    n2, k = size(X)
    @assert n == n2
    @assert length(b) == m

    @inbounds for j in 1:k
        for i in 1:m
            s = zero(eltype(A))
            @inbounds for t in 1:n
                s += A[i, t] * X[t, j]
            end
            s == b[i] || return false
        end
    end

    return true
end
