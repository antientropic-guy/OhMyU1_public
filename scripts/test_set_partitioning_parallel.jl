#!/usr/bin/env julia
# Run: julia --project=. --threads=4 scripts/test_set_partitioning_parallel.jl
# Compare serial vs concurrent runs, re-check files independently, then resume.
include("run_SetPartitioning.jl")
using Test
Threads.nthreads() >= 2 || error("Run with --threads=4 (at least 2 required)")
BLAS.set_num_threads(1)
mkpath(joinpath(PARTITION_ROOT,"tmp"))
test_root = mktempdir(joinpath(PARTITION_ROOT,"tmp");prefix="partition_parallel_",cleanup=false)
generate_partition_instances(test_root;ns=[12,13],instances=1:2,densities=["light"])
sp = partition_solver_params("ADD";smoke=true)

@testset "Frozen production parameters" begin
    @test PARTITION_ADD_STRATEGY == "best nonzero mixed"
    @test PARTITION_ADD_WEIGHT == 0.4
    for mode in ("EELS","ADD")
        p = partition_solver_params(mode)
        @test p.NUM_FEASIBLE_SAMPLES == (mode == "EELS" ? 400 : 1000)
        @test p.NUM_MPS_SAMPLES == 10000
        @test p.NUM_GLOBAL_ITER == 20
        @test p.NUM_SWEEP_ITER == p.LINK_DEGENERACY == 1
        @test p.LEARNING_RATE == p.UTILITY_FRACTION == 0.05
        @test p.KEEP_NUM_WORST == 0.0
    end
end

@testset "Serial / parallel input isolation and resume" begin
    for mode in ("EELS","ADD")
        serial_dir = joinpath(test_root,"serial")
        parallel_dir = partition_results_dir(test_root)
        run_partition_group(test_root,mode,"entropic","light",[12,13],1:2,sp,1;result_directory=serial_dir)
        run_partition_group(test_root,mode,"entropic","light",[12,13],1:2,sp,min(4,Threads.nthreads()))
        name = "res_$(mode)_entropic_light_test.jld2"
        serial, parallel = load(joinpath(serial_dir,name)),load(joinpath(parallel_dir,name))
        @test serial["configuration"] == parallel["configuration"]
        @test parallel["configuration"].protocol == 3
        @test parallel["configuration"].add_strategy == "best nonzero mixed"
        @test parallel["configuration"].add_weight == 0.4
        @test !parallel["configuration"].diversify_add && !parallel["configuration"].barrier
        for n in (12,13), i in 1:2
            input = load(partition_path(test_root,"light",n,i))
            p = input["instance"]
            # Independent expression, not the runner's objective closure.
            function independent_cost(x)
                o = input["objectives"]
                z = o.scenarios*x/(o.tau*sqrt(length(p.b)))
                t = maximum(z)
                dot(o.c,x)/sqrt(length(p.b)) + o.tau*(t+log(sum(exp.(z.-t))/length(z)))
            end
            da,db = serial["diagnostics"][(n,i)],parallel["diagnostics"][(n,i)]
            @test da.exact_cover_count == db.exact_cover_count == input["exact_cover_count"]
            @test da.initial_sha256 == db.initial_sha256
            @test da.input_sha256 == db.input_sha256 == parallel["configuration"].input_hashes[(n,i)]
            @test da.dp_states == db.dp_states == input["dp_states"]
            for arm in ("best_cost", mode == "EELS" ? "EELS" : PARTITION_ADD_STRATEGY)
                function branch(d)
                    b = d["res_dict"][arm]
                    arm == PARTITION_ADD_STRATEGY ? b[PARTITION_ADD_WEIGHT][n][i] : b[n][i]
                end
                a,b = branch(serial),branch(parallel)
                @test a.incub == b.incub
                @test a.learning_curve == b.learning_curve
                @test a.utility_curve == b.utility_curve
                @test a.c_min == b.c_min
                @test all(p.A*b.incub .== p.b)
                @test isapprox(independent_cost(b.incub),b.c_min;atol=1e-10,rtol=1e-10)
                @test da.arms[arm].initial_training_sha256 == db.arms[arm].initial_training_sha256
                if mode == "ADD"
                    @test da.arms[arm].external_first32_sha256 == db.arms[arm].external_first32_sha256
                    @test da.arms[arm].external_sample_counts == db.arms[arm].external_sample_counts
                else
                    @test da.arms[arm].evaluations == db.arms[arm].evaluations
                end
            end
        end
        before = sha256(read(joinpath(parallel_dir,name)))
        run_partition_group(test_root,mode,"entropic","light",[12,13],1:2,sp,min(4,Threads.nthreads()))
        @test sha256(read(joinpath(parallel_dir,name))) == before
    end
    wrong = load(partition_path(test_root,"light",12,1))
    @test_throws ErrorException run_partition_pair(wrong,"ADD","entropic","light",12,2,sp,"wrong")
end
println("Regression artifacts: ",test_root)
