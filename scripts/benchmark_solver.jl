using LinearAlgebra
using OhMyU1
using Random
using Statistics

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const NUM_GLOBAL_ITER = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const NUM_REPEATS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const SEED = 100

N >= 5 || throw(ArgumentError("N must be at least 5 with the default solver sample counts"))

A = OhMyU1.assignment_matrix(N)
b = ones(Int, 2N)
Q = OhMyU1.random_bilinear_form(Random.Xoshiro(SEED), N^2, 1.0, 7)
problem = OhMyU1.OptimizationProblem(
    A=A,
    b=b,
    cost_function=x -> dot(x, Q * x),
)
sampler! = (rng, destination) ->
    OhMyU1.fill_assignment_vectors!(rng, destination, N)
params = OhMyU1.SolverParams(
    NUM_GLOBAL_ITER=NUM_GLOBAL_ITER,
    KEEP_NUM_WORST=0.0,
    LEARNING_RATE=0.05,
)

function run_benchmark()
    return OhMyU1.solve(
        problem, sampler!, params, "weighted rank";
        diversify=false,
        barrier=false,
        parallel=true,
        print_stats=false,
        rank_weight=0.1,
        seed=SEED,
    )
end

println("Warming up n=$N, iterations=$NUM_GLOBAL_ITER ...")
run_benchmark()

times = Float64[]
bytes = Int[]
for repeat in 1:NUM_REPEATS
    GC.gc()
    measurement = @timed run_benchmark()
    push!(times, measurement.time)
    push!(bytes, measurement.bytes)
    println("run $repeat: $(round(measurement.time; digits=3)) s, " *
            "$(round(measurement.bytes / 2.0^30; digits=3)) GiB allocated")
end

println("median: $(round(median(times); digits=3)) s")
