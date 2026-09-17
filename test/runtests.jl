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
