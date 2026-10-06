using Test
include(joinpath(@__DIR__, "run_EELS_optimization.jl"))
BLAS.set_num_threads(1)

@testset "Assignment RNG: all 500 production seeds" begin
    jobs = [(objective, n, i) for objective in ("linear", "quadratic") for n in 12:16 for i in 1:50]
    function generate_initial(job; interleave=false)
        objective, n, i = job
        Random.seed!(pair_seed("test", objective, n, i))
        interleave && yield() # another task can consume its own RNG here
        initial = zeros(Int, n^2, 400)
        fill_assignment_vectors!(initial, n) # actual production sampler
        initial
    end
    # Compare complete matrices, not just minimum costs or summary statistics.
    reference = [generate_initial(job) for job in jobs]
    tasks = [Threads.@spawn begin
        rand(37) # unrelated random activity before reseeding this task
        generated = generate_initial(job; interleave=true)
        generated == expected
    end for (job, expected) in zip(jobs, reference)]
    @test all(fetch.(tasks))
    @test length(unique(pair_seed("test", o, n, i) for (o, n, i) in jobs)) == 500
    # A changed seed actually changes the data, rather than being ignored.
    @test reference[1] != reference[2]
    @info "Production initial matrices match serial/interleaved generation" jobs=length(jobs) threads=Threads.nthreads()
end

@testset "EELS scarce-data objective experiment" begin
    sp = SolverParams(NUM_FEASIBLE_SAMPLES=20, NUM_MPS_SAMPLES=40,
        NUM_GLOBAL_ITER=2, KEEP_NUM_WORST=0.0)
    @test_throws ErrorException instance_path("train", "linear", 12, 1)
    for objective in ("linear", "quadratic"), n in 12:16, i in 1:50
        @test isfile(instance_path("test", objective, n, i))
    end
    mktempdir() do output
        # Two jobs exercise the checkpoint lock and worker scheduling.
        path = run_group("test", "linear", 12:12, 1:2, sp, output)
        saved = load(path)
        @test saved["res_dict"] isa ResultDict
        for i in 1:2
            baseline = saved["res_dict"]["best_cost"][12][i]
            eels = saved["res_dict"]["EELS"][12][i]
            @test baseline.learning_curve[1] == eels.learning_curve[1]
            @test saved["diagnostics"]["best_cost"][(12, i)].initial_training_sha256 ==
                  saved["diagnostics"]["EELS"][(12, i)].initial_training_sha256
            c = load(instance_path("test", "linear", 12, i), "c")
            for arm in ARMS
                stats = saved["res_dict"][arm][12][i]
                details = saved["diagnostics"][arm][(12, i)]
                @test length(stats.learning_curve) == 3
                @test length(stats.utility_curve) == 3
                @test all(diff(stats.learning_curve) .<= 0)
                @test all(isfinite, stats.learning_curve)
                @test assignment_matrix(12) * stats.incub == ones(Int, 24)
                @test dot(c, stats.incub) ≈ stats.c_min
                @test details.evaluations[end] <= 20 + 2 * 40
                @test all(details.training_sizes .<= 20)
            end
        end
        before = read(path)
        run_group("test", "linear", 12:12, 1:2, sp, output)
        @test read(path) == before # resume does not rerun completed jobs
        changed = SolverParams(NUM_FEASIBLE_SAMPLES=21, NUM_MPS_SAMPLES=40,
            NUM_GLOBAL_ITER=2, KEEP_NUM_WORST=0.0)
        @test_throws ErrorException run_group("test", "linear", 12:12, 1:2, changed, output)
    end

    # Equal-cost/collapsed data must not produce a zero-temperature division.
    A = assignment_matrix(3)
    initial = zeros(Int, 9, 20)
    Random.seed!(123)
    fill_assignment_vectors!(initial, 3)
    problem = OptimizationProblem(A=A, b=ones(Int, 6), cost_function=x -> 1.0)
    first, _ = run_arm(problem, initial, sp, 123; eels=true)
    second, _ = run_arm(problem, initial, sp, 123; eels=true)
    @test first.learning_curve == second.learning_curve == [1.0, 1.0, 1.0]
    @test first.incub == second.incub

    # Capture the matrices that the two real optimization runs actually use,
    # after the first arm has trained, sampled and changed the task RNG state.
    n = 12
    Random.seed!(pair_seed("test", "quadratic", n, 1))
    initial = zeros(Int, n^2, 400)
    fill_assignment_vectors!(initial, n)
    untouched = copy(initial)
    q = load(instance_path("test", "quadratic", n, 1), "q")
    problem = OptimizationProblem(A=assignment_matrix(n), b=ones(Int, 2n), cost_function=x -> dot(x, q*x))
    observed = Dict{String,Matrix{Int}}()
    for arm in ARMS
        run_arm(problem, initial, sp, 123; eels=(arm == "EELS"),
            on_initial_training=T -> (observed[arm] = T))
        @test initial == untouched
        rand(1000) # deliberately disturb the current RNG between the arms
    end
    @test observed["best_cost"] == observed["EELS"]
    @test size(observed["best_cost"], 2) == size(unique(initial; dims=2), 2)
end
