#!/usr/bin/env julia

# Generate an untouched confirmatory benchmark set. These files deliberately go
# into test_* directories so they cannot be confused with the instances already
# used to inspect/tune w. Existing files are never overwritten.

using JLD2
using Random
using SparseArrays

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LINEAR_DIR = joinpath(ROOT, "data", "test_random_vectors")
const QUADRATIC_DIR = joinpath(ROOT, "data", "test_random_bilinear_forms")

function quadratic_form(rng::AbstractRNG, N::Int, r::Real)
    # Same construction as src/tools.jl: uniform entries followed by
    # symmetrization. Consequently diagonal entries are uniform; off-diagonal
    # entries are averages of two uniforms (a symmetric triangular law).
    raw = rand(rng, N, N) .* (2r) .- r
    return sparse(0.5 .* (raw .+ raw'))
end

function generate_test_instances(; n_sizes=12:16, instances=1:50,
                                 linear_radius=50.0, quadratic_radius=7.0,
                                 seed=0x4f684d795531)
    mkpath(LINEAR_DIR)
    mkpath(QUADRATIC_DIR)
    rng = MersenneTwister(seed)
    generated = 0
    for n in n_sizes, i in instances
        c_path = joinpath(LINEAR_DIR, "c_$(n)_$(i)_r=$(Int(linear_radius))_test.jld2")
        q_path = joinpath(QUADRATIC_DIR, "q_$(n)_$(i)_r=$(Int(quadratic_radius))_test.jld2")
        (ispath(c_path) || ispath(q_path)) &&
            error("Refusing to overwrite test instance n=$n, i=$i")
        c = rand(rng, n^2) .* (2linear_radius) .- linear_radius
        q = quadratic_form(rng, n^2, quadratic_radius)
        metadata = (split="test", seed=string(seed), n=n, instance=i)
        jldsave(c_path; c, metadata)
        jldsave(q_path; q, metadata)
        generated += 1
    end
    println("Generated $generated paired linear/quadratic test instances.")
    println("Linear:    $LINEAR_DIR")
    println("Quadratic: $QUADRATIC_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_test_instances()
end
