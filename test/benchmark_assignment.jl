# Diagnostic benchmark, separate from the regression-test runner.
# Run one process at a time with --threads=1,1 and --threads=4,1.
include("helpers.jl")
include("large_performance.jl")
using Profile
output = isempty(ARGS) ? joinpath(@__DIR__,"output","optimization_$(Threads.nthreads()).toml") : abspath(ARGS[1])
mkpath(dirname(output))
parallel = Threads.nthreads()>1
result = Dict{String,Any}("workload"=>LARGE_WORKLOAD,
    "cpu"=>Sys.cpu_info()[1].model,"julia"=>string(VERSION),
    "threads"=>Threads.nthreads(),"interactive_threads"=>Threads.nthreads(:interactive),
    "blas_threads"=>BLAS.get_num_threads(),"parallel_sampling"=>parallel,
    "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(ROOT,"src",f)))) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl")))
for strategy in ("best_cost","best nonzero mixed")
    large_benchmark_run(parallel;iterations=1,strategy)
    times = Float64[]; bytes = Int[]; costs = Float64[]; gcs = Float64[]
    for rep in 1:3
        GC.gc()
        r = @timed large_benchmark_run(parallel;strategy)
        push!(times,r.time); push!(bytes,r.bytes); push!(costs,r.value.c_min); push!(gcs,r.gctime)
        println(strategy," repetition=",rep," seconds=",r.time," bytes=",r.bytes," cost=",r.value.c_min)
        flush(stdout)
    end
    result[strategy] = Dict("seconds"=>median(times),"bytes"=>median(bytes),"times"=>times,"costs"=>costs,"gc_seconds"=>gcs)
end
result["max_rss_bytes"] = Sys.maxrss()
open(io->TOML.print(io,result),output,"w")
Profile.clear()
Profile.@profile large_benchmark_run(parallel;strategy="best nonzero mixed")
open(output*".profile.txt","w") do io
    Profile.print(IOContext(io,:displaysize=>(20000,200));format=:flat,sortedby=:count,mincount=30)
end
