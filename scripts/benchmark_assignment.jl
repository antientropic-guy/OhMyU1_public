#!/usr/bin/env julia

"""
Fair one-iteration benchmark of OhMyU1 and ConstrainTNet on assignment problems.

The timed region excludes construction/canonicalization of the initial MPS and
contains the same work for both packages:

  * one right/left training sweep over the same `num_samples` rows;
  * one batch of exactly `num_samples` samples from the trained MPS.

ConstrainTNet.optimizer is deliberately not timed directly: for one loop it
calls `sample` 400 times before training and 400 times after training, whereas
the Scaling.ipynb OhMyU1 wrapper receives its training set from outside and
samples only 400 times.  Timing the public training and sampling kernels
separately removes that mismatch without changing either implementation.

Run from anywhere:

    julia --project=. scripts/benchmark_assignment.jl

Useful quick check:

    julia --project=. scripts/benchmark_assignment.jl --n=3 --samples=16 --repeats=1
"""

using LinearAlgebra
using Printf
using Random
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const CT_ROOT = joinpath(REPO_ROOT, "ConstrainTNet.jl")
const RESULT_PREFIX = "BENCH_RESULT|"
const PROCESS_BACKEND = let
    argument = findfirst(arg -> startswith(arg, "--backend="), ARGS)
    isnothing(argument) ? "driver" : split(ARGS[argument], '='; limit=2)[2]
end

# Each child process has only its own package environment active. Imports must
# be at top level in Julia, hence this dispatch happens before main().
if PROCESS_BACKEND == "ohmy"
    using OhMyU1
    import OhMyU1: TrainParams, compute_indices, compute_site_charges,
        init_u1_mps, normalize!, orthogonalize!, sample_nondeg!,
        sample_nondeg_parallel!, train_nondeg!, u1_norm
elseif PROCESS_BACKEND == "constraintnet"
    using ConstrainTNet
    import ConstrainTNet: TrainParams, constrained_orthogonalize!,
        constraints_to_mps, normalize!, training
    import ITensors: maxlinkdim, sample
end

function parse_options(args)
    opts = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unknown positional argument: $arg")
        key_value = split(arg[3:end], "="; limit=2)
        length(key_value) == 2 || error("Options must have the form --name=value: $arg")
        opts[key_value[1]] = key_value[2]
    end
    return opts
end

parse_int_list(s) = parse.(Int, split(s, ','))

function assignment_matrix(n::Int)
    A = zeros(Int, 2n, n^2)
    for worker in 1:n, job in 1:n
        variable = (worker - 1) * n + job
        A[job, variable] = 1
        A[n + worker, variable] = 1
    end
    return A
end

"""Generate identical candidate and Boltzmann-resampled training data in either environment."""
function benchmark_data(n::Int, num_samples::Int, seed::Int)
    rng = MersenneTwister(seed + 10_000n)
    num_variables = n^2
    candidates = zeros(Int, num_variables, num_samples)
    for sample_index in axes(candidates, 2)
        permutation = randperm(rng, n)
        for worker in 1:n
            candidates[(worker - 1) * n + permutation[worker], sample_index] = 1
        end
    end

    Q = rand(rng, -7:7, num_variables, num_variables)
    costs = [dot(view(candidates, :, j), Q * view(candidates, :, j)) for j in axes(candidates, 2)]
    temperature = std(costs)
    if iszero(temperature) || !isfinite(temperature)
        weights = ones(Float64, num_samples)
    else
        weights = exp.(-(costs .- minimum(costs)) ./ temperature)
    end
    cumulative = cumsum(weights ./ sum(weights))

    # Resampling makes the empirical training distribution identical for both
    # APIs. OhMyU1 then receives uniform probabilities for these (possibly
    # repeated) columns, exactly as ConstrainTNet.Dataset does for its rows.
    training = similar(candidates)
    for j in axes(training, 2)
        source = searchsortedfirst(cumulative, rand(rng))
        training[:, j] .= candidates[:, source]
    end
    return Q, training
end

function input_checksum(Q, training)
    qsum = sum((i + 17j) * Q[i, j] for i in axes(Q, 1), j in axes(Q, 2))
    tsum = sum((i + 31j) * training[i, j] for i in axes(training, 1), j in axes(training, 2))
    return string(qsum, ':', tsum)
end

function median_elapsed(operation, repeats::Int, prepare)
    times = Vector{Float64}(undef, repeats)
    for repetition in 1:repeats
        state = prepare(repetition)
        GC.gc()
        times[repetition] = @elapsed operation(state, repetition)
    end
    return median(times)
end

function result_line(values)
    fields = (string(key, '=', value) for (key, value) in pairs(values))
    println(RESULT_PREFIX, join(fields, '|'))
end

function exact_assignment_mps_ohmy(A, b)
    num_constraints, num_variables = size(A)
    n = isqrt(num_variables)
    n^2 == num_variables || error("assignment benchmark expects n^2 variables")
    num_constraints == 2n || error("assignment benchmark expects 2n constraints")
    n < 8sizeof(Int) - 1 || error("n is too large for bit-mask charge enumeration")

    # Exact residual charges at every cut, specialized to the worker-major
    # variable order used by assignment_matrix. A charge is determined by the
    # subset of jobs already used, whether the current worker has already been
    # assigned in the processed part of its row, and the completed/future rows.
    # This enumerates at most 2^n masks per cut rather than n! assignments.
    tuple_type = typeof(Tuple(b))
    valid = Vector{Set{tuple_type}}(undef, num_variables + 1)
    for processed in 0:num_variables
        completed_workers, processed_in_row = divrem(processed, n)
        charges = Set{tuple_type}()
        for mask in 0:(Int(1) << n) - 1
            used_count = count_ones(mask)
            no_current_selection = used_count == completed_workers &&
                (processed_in_row == 0 || any(job -> iszero(mask & (Int(1) << (job - 1))), processed_in_row + 1:n))
            current_selected = processed_in_row > 0 &&
                used_count == completed_workers + 1 &&
                any(job -> !iszero(mask & (Int(1) << (job - 1))), 1:processed_in_row)
            (no_current_selection || current_selected) || continue

            job_residuals = ntuple(job -> 1 - Int(!iszero(mask & (Int(1) << (job - 1)))), n)
            worker_residuals = ntuple(worker -> begin
                worker <= completed_workers && return 0
                worker == completed_workers + 1 && processed_in_row > 0 && current_selected && return 0
                return 1
            end, n)
            push!(charges, (job_residuals..., worker_residuals...))
        end
        valid[processed + 1] = charges
    end

    valid[1] == Set([Tuple(b)]) || error("invalid left boundary charge")
    valid[end] == Set([ntuple(_ -> 0, num_constraints)]) || error("invalid right boundary charge")

    max_charges = maximum(length, valid)
    link_charges = zeros(Int, num_variables + 1, num_constraints, max_charges)
    for link in eachindex(valid)
        charges = collect(valid[link])
        for j in 1:max_charges
            link_charges[link, :, j] .= charges[mod1(j, length(charges))]
        end
    end

    domains = [[0, 1] for _ in 1:num_variables]
    site_charges = compute_site_charges(A, domains)
    link_indices, site_indices = compute_indices(link_charges, site_charges, b, domains, A)
    mps = init_u1_mps(Float64, link_indices, site_indices, A, b, 1, fill(2, num_variables))
    return mps, max_charges
end

function run_ohmy(ns, num_samples, repeats, seed)
    for n in ns
        A = assignment_matrix(n)
        b = ones(Int, 2n)
        Q, training = benchmark_data(n, num_samples, seed)
        all(A * training .== b) || error("infeasible shared training data for n=$n")

        construction_s = @elapsed begin
            initial_mps, max_support = exact_assignment_mps_ohmy(A, b)
            orthogonalize!(initial_mps)
            norm2 = u1_norm(initial_mps)^2
            normalize!(initial_mps)
        end
        expected_norm2 = Float64(factorial(big(n)))
        norm_relerr = abs(norm2 - expected_norm2) / expected_norm2
        norm_relerr <= 1e-10 || error("OhMyU1 initial MPS is not the exact uniform assignment MPS for n=$n")

        probabilities = fill(1.0 / num_samples, num_samples)
        train_params = TrainParams(0.05, 10^4)

        # Untimed calls remove JIT compilation from all reported measurements.
        warm = deepcopy(initial_mps)
        train_nondeg!(warm, 1, training, probabilities, train_params)
        warm_buffer = zeros(Int, n^2, min(num_samples, 2))
        sample_nondeg!(warm, warm_buffer, size(warm_buffer, 2))
        sample_nondeg_parallel!(warm, warm_buffer, size(warm_buffer, 2))

        train_s = median_elapsed(repeats, _ -> deepcopy(initial_mps)) do mps, repetition
            Random.seed!(seed + repetition)
            train_nondeg!(mps, 1, training, probabilities, train_params)
        end

        trained = deepcopy(initial_mps)
        train_nondeg!(trained, 1, training, probabilities, train_params)
        serial_buffer = zeros(Int, n^2, num_samples)
        sample_serial_s = median_elapsed(repeats, _ -> deepcopy(trained)) do mps, repetition
            Random.seed!(seed + repetition)
            sample_nondeg!(mps, serial_buffer, num_samples)
        end
        all(A * serial_buffer .== b) || error("OhMyU1 produced an infeasible sample for n=$n")

        parallel_buffer = similar(serial_buffer)
        sample_native_s = median_elapsed(repeats, _ -> deepcopy(trained)) do mps, repetition
            Random.seed!(seed + repetition)
            sample_nondeg_parallel!(mps, parallel_buffer, num_samples)
        end
        all(A * parallel_buffer .== b) || error("OhMyU1 produced an infeasible parallel sample for n=$n")

        result_line((backend="OhMyU1", n=n, variables=n^2, constraints=2n,
            samples=num_samples, sweeps=1, repeats=repeats, threads=Threads.nthreads(),
            checksum=input_checksum(Q, training), construction_s=construction_s,
            norm2=norm2, norm_relerr=norm_relerr, max_support=max_support,
            train_s=train_s, sample_serial_s=sample_serial_s,
            sample_native_s=sample_native_s,
            balanced_serial_s=train_s + sample_serial_s,
            balanced_native_s=train_s + sample_native_s))
    end
end

function run_constraintnet(ns, num_samples, repeats, seed)
    for n in ns
        A = assignment_matrix(n)
        b = ones(Int, 2n)
        Q, training_columns = benchmark_data(n, num_samples, seed)
        all(A * training_columns .== b) || error("infeasible shared training data for n=$n")
        training_rows = Matrix(transpose(training_columns))
        flux_mps = QRegion([Box(b, b)])

        construction_s = @elapsed prepared = redirect_stdout(devnull) do
            mps, backward_indices, forward_indices, _ = constraints_to_mps(
                A, b, b; verbose=false, flux_center=1, block_dim=1)
            constrained_orthogonalize!(mps, 1;
                left_canonical_indices=forward_indices,
                right_canonical_indices=backward_indices,
                flux_mps=flux_mps, min_blockdim=0, verbose=false)
            mps_norm2 = norm(mps)^2
            support = maxlinkdim(mps)
            normalize!(mps)
            return mps, backward_indices, forward_indices, support, mps_norm2
        end
        initial_mps, backward, forward, max_support, norm2 = prepared
        expected_norm2 = Float64(factorial(big(n)))
        norm_relerr = abs(norm2 - expected_norm2) / expected_norm2
        norm_relerr <= 1e-10 || error("ConstrainTNet initial MPS is not the exact uniform assignment MPS for n=$n")
        train_params = TrainParams(Float32(0.05), 1)
        kwargs = (left_canonical_indices=forward, right_canonical_indices=backward,
            flux_mps=flux_mps, min_blockdim=0, verbose=false)

        # Untimed calls remove JIT compilation from all reported measurements.
        warm = deepcopy(initial_mps)
        _, warm = training(train_params, warm, training_rows; kwargs...)
        sample(warm)

        train_s = median_elapsed(repeats, _ -> deepcopy(initial_mps)) do mps, repetition
            Random.seed!(seed + repetition)
            training(train_params, mps, training_rows; kwargs...)
        end

        trained = deepcopy(initial_mps)
        _, trained = training(train_params, trained, training_rows; kwargs...)
        last_samples = zeros(Int, n^2, num_samples)
        sample_s = median_elapsed(repeats, _ -> deepcopy(trained)) do mps, repetition
            Random.seed!(seed + repetition)
            for j in 1:num_samples
                last_samples[:, j] .= sample(mps) .- 1
            end
        end
        all(A * last_samples .== b) || error("ConstrainTNet produced an infeasible sample for n=$n")

        result_line((backend="ConstrainTNet", n=n, variables=n^2, constraints=2n,
            samples=num_samples, sweeps=1, repeats=repeats, threads=Threads.nthreads(),
            checksum=input_checksum(Q, training_columns), construction_s=construction_s,
            norm2=norm2, norm_relerr=norm_relerr, max_support=max_support,
            train_s=train_s, sample_serial_s=sample_s, sample_native_s=sample_s,
            balanced_serial_s=train_s + sample_s, balanced_native_s=train_s + sample_s))
    end
end

function parse_result(output::String)
    rows = Vector{Dict{String,String}}()
    for line in split(output, '\n')
        startswith(line, RESULT_PREFIX) || continue
        fields = split(line[length(RESULT_PREFIX)+1:end], '|')
        row = Dict{String,String}()
        for field in fields
            key, value = split(field, '='; limit=2)
            row[key] = value
        end
        push!(rows, row)
    end
    return rows
end

function backend_command(project, script, backend, ns, num_samples, repeats, seed)
    julia = Base.julia_cmd()
    n_arg = join(ns, ',')
    return `$julia --startup-file=no --project=$project $script --backend=$backend --n=$n_arg --samples=$num_samples --repeats=$repeats --seed=$seed`
end

function write_csv(path, rows)
    columns = ["backend", "n", "variables", "constraints", "samples", "sweeps", "repeats",
        "threads", "checksum", "construction_s", "norm2", "norm_relerr", "max_support",
        "train_s", "sample_serial_s", "sample_native_s", "balanced_serial_s", "balanced_native_s"]
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join((row[column] for column in columns), ','))
        end
    end
end

function print_summary(rows)
    by_key = Dict((row["backend"], parse(Int, row["n"])) => row for row in rows)
    ns = sort(unique(parse(Int, row["n"]) for row in rows))
    println("\nBalanced one-sweep benchmark (MPS construction excluded)")
    println("n   OhMyU1 serial   OhMyU1 native   ConstrainTNet   speedup(serial)   speedup(native)")
    for n in ns
        oh = by_key[("OhMyU1", n)]
        ct = by_key[("ConstrainTNet", n)]
        oh_serial = parse(Float64, oh["balanced_serial_s"])
        oh_native = parse(Float64, oh["balanced_native_s"])
        ct_time = parse(Float64, ct["balanced_serial_s"])
        @printf("%-3d %14.6f %16.6f %15.6f %17.2fx %16.2fx\n",
            n, oh_serial, oh_native, ct_time, ct_time / oh_serial, ct_time / oh_native)
    end
end

function main(args)
    opts = parse_options(args)
    ns = parse_int_list(get(opts, "n", "4,5,6,7,8,9,10"))
    num_samples = parse(Int, get(opts, "samples", "400"))
    repeats = parse(Int, get(opts, "repeats", "3"))
    seed = parse(Int, get(opts, "seed", "20260923"))
    backend = get(opts, "backend", "driver")

    all(>(1), ns) || error("all n values must exceed 1")
    num_samples > 0 || error("--samples must be positive")
    repeats > 0 || error("--repeats must be positive")

    if backend == "ohmy"
        run_ohmy(ns, num_samples, repeats, seed)
        return
    elseif backend == "constraintnet"
        run_constraintnet(ns, num_samples, repeats, seed)
        return
    elseif backend != "driver"
        error("unknown backend: $backend")
    end

    output_path = abspath(get(opts, "output", joinpath(REPO_ROOT, "assignment_benchmark.csv")))
    script = abspath(@__FILE__)
    rows = Vector{Dict{String,String}}()
    for (name, project) in (("ohmy", REPO_ROOT), ("constraintnet", CT_ROOT))
        command = backend_command(project, script, name, ns, num_samples, repeats, seed)
        println("Running $name backend with project $project")
        # Julia 1.12/Windows cannot currently attach an IOBuffer (or the devnull
        # used by `read(command, String)`) to this child process reliably. A
        # temporary regular file avoids that platform-specific EBADF failure.
        output = mktemp() do path, io
            close(io)
            run(pipeline(command; stdout=path))
            read(path, String)
        end
        print(output)
        append!(rows, parse_result(output))
    end

    length(rows) == 2length(ns) || error("expected $(2length(ns)) result rows, got $(length(rows))")
    for n in ns
        matching = filter(row -> parse(Int, row["n"]) == n, rows)
        length(unique(row["checksum"] for row in matching)) == 1 || error("backend inputs differ for n=$n")
    end
    write_csv(output_path, rows)
    print_summary(rows)
    println("\nRaw measurements: $output_path")
end

main(ARGS)
