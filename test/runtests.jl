using Test
using LinearAlgebra
using OhMyU1
using Random

@testset "deterministic random helpers" begin
    n = 4
    x1 = zeros(Int, n^2, 200)
    x2 = similar(x1)
    OhMyU1.fill_assignment_vectors_parallel!(Random.Xoshiro(17), x1, n)
    OhMyU1.fill_assignment_vectors_parallel!(Random.Xoshiro(17), x2, n)
    @test x1 == x2
    @test all(sum(x1; dims=1) .== n)

    legacy_sampler = destination ->
        OhMyU1.fill_assignment_vectors_parallel!(destination, n)
    legacy1 = similar(x1)
    legacy2 = similar(x1)
    OhMyU1._call_feasible_sampler!(legacy_sampler, Random.Xoshiro(22), legacy1)
    OhMyU1._call_feasible_sampler!(legacy_sampler, Random.Xoshiro(22), legacy2)
    @test legacy1 == legacy2

    q1 = OhMyU1.random_bilinear_form(Random.Xoshiro(18), 20, 0.4, 7)
    q2 = OhMyU1.random_bilinear_form(Random.Xoshiro(18), 20, 0.4, 7)
    @test q1 == q2
end

@testset "MPS sampling is schedule-independent" begin
    n = 4
    A = OhMyU1.assignment_matrix(n)
    b = ones(Int, 2n)
    training = zeros(Int, n^2, 48)
    OhMyU1.fill_assignment_vectors!(Random.Xoshiro(19), training, n)
    problem = OhMyU1.OptimizationProblem(A=A, b=b, cost_function=sum)
    mps = OhMyU1.build_mps_from_feasible_samples(problem, training, 1, false)
    OhMyU1.orthogonalize!(mps)
    OhMyU1.normalize!(mps)

    serial = zeros(Int, n^2, 200)
    parallel = similar(serial)
    OhMyU1.sample_nondeg!(Random.Xoshiro(20), mps, serial, 200)
    OhMyU1.sample_nondeg_parallel!(Random.Xoshiro(20), mps, parallel, 200)
    @test serial == parallel
    @test A * serial == repeat(b, 1, size(serial, 2))
end

@testset "solve is repeatable" begin
    n = 4
    A = OhMyU1.assignment_matrix(n)
    b = ones(Int, 2n)
    c = rand(Random.Xoshiro(21), n^2)
    problem = OhMyU1.OptimizationProblem(A=A, b=b, cost_function=x -> dot(c, x))
    params = OhMyU1.SolverParams(NUM_FEASIBLE_SAMPLES=40,
                                 NUM_MPS_SAMPLES=80,
                                 NUM_GLOBAL_ITER=2,
                                 UTILITY_FRACTION=0.2,
                                 KEEP_NUM_WORST=0.0)
    sampler = (rng, destination) ->
        OhMyU1.fill_assignment_vectors_parallel!(rng, destination, n)

    run1 = OhMyU1.solve(problem, sampler, params, "best_cost";
                        seed=20260917, parallel=true, print_stats=false)
    run2 = OhMyU1.solve(problem, sampler, params, "best_cost";
                        seed=20260917, parallel=true, print_stats=false)
    serial = OhMyU1.solve(problem, sampler, params, "best_cost";
                          seed=20260917, parallel=false, print_stats=false)

    for field in fieldnames(typeof(run1))
        @test isequal(getfield(run1, field), getfield(run2, field))
        @test isequal(getfield(run1, field), getfield(serial, field))
    end
end

@testset "tiny dominant SVD agrees with LAPACK" begin
    rng = Random.Xoshiro(24)
    for (m, n) in ((1, 1), (1, 3), (3, 1), (2, 2), (3, 4))
        block = randn(rng, m, n)
        u, sigma, v = OhMyU1._dominant_singular_triplet(block)
        reference = svd(block; full=false)
        @test sigma ≈ reference.S[1] rtol=1e-12 atol=1e-12
        @test sigma .* (u * v') ≈
              reference.S[1] .* (reference.U[:, 1] * reference.V[:, 1]') rtol=1e-11 atol=1e-11
    end

    for block in (zeros(1, 3), zeros(3, 1), zeros(2, 2))
        u, sigma, v = OhMyU1._dominant_singular_triplet(block)
        @test sigma == 0
        @test sigma .* (u * v') == block
    end
end

@testset "non-degenerate orthogonalization preserves amplitudes" begin
    n = 4
    A = OhMyU1.assignment_matrix(n)
    b = ones(Int, 2n)
    training = zeros(Int, n^2, 48)
    OhMyU1.fill_assignment_vectors!(Random.Xoshiro(25), training, n)
    problem = OhMyU1.OptimizationProblem(A=A, b=b, cost_function=sum)
    mps = OhMyU1.build_mps_from_feasible_samples(problem, training, 1, false)
    amplitudes_before = [mps[column] for column in eachcol(training)]
    OhMyU1.orthogonalize!(mps)
    amplitudes_after = [mps[column] for column in eachcol(training)]
    @test amplitudes_after ≈ amplitudes_before rtol=1e-12 atol=1e-12

    matrix = OhMyU1.collect_right(mps.Cores[2])
    blocks_l, blocks_q = OhMyU1.u1_lq(matrix)
    for charge in keys(matrix.Blocks)
        @test blocks_l[charge] * blocks_q[charge] ≈ matrix.Blocks[charge]
    end
end

@testset "sparse graph distance agrees with dense implementation" begin
    n = 4
    A = OhMyU1.assignment_matrix(n)
    b = ones(Int, 2n)
    training = zeros(Int, n^2, 48)
    OhMyU1.fill_assignment_vectors!(Random.Xoshiro(26), training, n)
    problem = OhMyU1.OptimizationProblem(A=A, b=b, cost_function=sum)
    mps = OhMyU1.build_mps_from_feasible_samples(problem, training, 1, false)
    memorized = [Set{Vector{Int}}(mps.LinkIndices[i + 1].Charges)
                 for i in 1:size(A, 2)]
    update_rows, update_values = OhMyU1._constraint_column_updates(A)
    initial_charge = first(mps.LinkIndices[1].Charges)
    sparse_distances = OhMyU1._graph_distances_parallel(
        training, initial_charge, memorized, update_rows, update_values)
    dense_buffer = similar(initial_charge)
    dense_distances = [OhMyU1.graph_dist_global!(dense_buffer, column, mps, memorized)
                       for column in eachcol(training)]
    @test sparse_distances == dense_distances
end
