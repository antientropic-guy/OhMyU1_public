#!/usr/bin/env julia
# Lightweight parent: expensive library imports happen only in workers.
using TOML, SHA
const ROOT=normpath(joinpath(@__DIR__,".."))
const OUT=joinpath(@__DIR__,"output");mkpath(OUT)
isfile(joinpath(OUT,"summary.toml")) && rm(joinpath(OUT,"summary.toml"))
const localbaseline=joinpath(@__DIR__,"local_performance_baseline.toml")
const baseline=isfile(localbaseline) ? localbaseline : joinpath(@__DIR__,"fixtures","performance_reference.toml")
const hardware=Dict("cpu_model"=>Sys.cpu_info()[1].model,"logical_cpus"=>Sys.CPU_THREADS,"machine"=>Sys.MACHINE,"julia"=>string(VERSION))
const workload="assignment5_initial32_samples512_iterations3_seed6021_v1"
all(a->a in ("--calibrate","--large-only"),ARGS) || error("Usage: [--calibrate | --large-only]")
large_only="--large-only" in ARGS
large_only && "--calibrate" in ARGS && error("Calibration requires the complete correctness suite")
calibrate="--calibrate" in ARGS
started=time();passed=true;measurements=Dict{String,Any}()
# Verify the requested no-algorithm-edits invariant during the run.
sourcehash()=Dict(f=>bytes2hex(sha256(read(joinpath(ROOT,"src",f)))) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
before=sourcehash()
if !large_only
    for threads in (1,4)
        local report, log, cmd, proc
        report=joinpath(OUT,"threads_$threads.toml");log=joinpath(OUT,"threads_$threads.log")
        isfile(report) && rm(report) # Never consume a stale report after a worker crash.
        extra=threads==4 ? ["--threaded"] : String[]
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$ROOT --threads=$threads,1 $(joinpath(@__DIR__,"worker.jl")) $report $extra`
        println("Testing $threads default thread(s) + one interactive thread...");flush(stdout)
        proc=open(log,"w") do io;run(pipeline(ignorestatus(cmd),stdout=io,stderr=io));end
        global passed &= success(proc)
        if isfile(report);measurements[string(threads)]=TOML.parsefile(report);end
        println(read(log,String));flush(stdout)
    end
    log=joinpath(OUT,"no_interactive_pool.log")
    cmd=`$(Base.julia_cmd()) --startup-file=no --project=$ROOT --threads=4,0 $(joinpath(@__DIR__,"thread_layout.jl"))`
    proc=open(log,"w") do io;run(pipeline(ignorestatus(cmd),stdout=io,stderr=io));end
    passed &= success(proc);println(read(log,String))
    @assert sourcehash()==before "Tests modified algorithm source files"
    if haskey(measurements,"1") && haskey(measurements,"4") && all(haskey(measurements[k],"parallel") for k in ("1","4"))
        a=measurements["1"];b=measurements["4"]
        # Threading a small workload is not required to speed it up. Detect large regressions.
        for mode in ("serial","parallel")
            time_ok=b[mode]["seconds"]<=max(.1,2.5a[mode]["seconds"])
            mem_ok=b[mode]["allocated_bytes"]<=2a[mode]["allocated_bytes"]+20_000_000
            println("Cross-thread $mode: time=$time_ok, allocations=$mem_ok")
            global passed &= time_ok && mem_ok
        end
        rss_ok=b["max_rss_bytes"]<=1.5a["max_rss_bytes"]+128_000_000
        println("Cross-thread peak RSS: $rss_ok");global passed &= rss_ok
        if isfile(baseline)
            ref=TOML.parsefile(baseline)
            same=get(ref,"hardware",nothing)==hardware && get(ref,"workload",nothing)==workload
            if !same
                println("Local baseline belongs to another CPU/Julia; refusing a misleading timing comparison.")
                isfile(localbaseline) && !calibrate && (global passed=false)
            elseif !calibrate
                for k in ("1","4"),mode in ("serial","parallel")
                    now=measurements[k][mode];old=ref[k][mode]
                    ok=now["seconds"]<=1.5old["seconds"]+.02 && now["allocated_bytes"]<=1.25old["allocated_bytes"]+1_000_000
                    println("Local baseline $k/$mode: $ok");global passed &= ok
                    global passed &= measurements[k]["max_rss_bytes"]<=1.25ref[k]["max_rss_bytes"]+64_000_000
                end
            end
        else
            println("No local baseline: portable ceilings and cross-thread comparison only. Use --calibrate after correctness passes.")
        end
    end
end
quick_wall_seconds=time()-started
println("Small-workload and correctness stage: $(round(quick_wall_seconds;digits=1)) seconds")

large_measurements=Dict{String,Any}()
for threads in (1,4)
    local report, log, cmd, proc
    report=joinpath(OUT,"large_threads_$threads.toml");log=joinpath(OUT,"large_threads_$threads.log")
    isfile(report) && rm(report)
    cmd=`$(Base.julia_cmd()) --startup-file=no --project=$ROOT --threads=$threads,1 $(joinpath(@__DIR__,"large_worker.jl")) $report`
    println("Testing assignment 10x10 with $threads default thread(s)...");flush(stdout)
    proc=open(log,"w") do io;run(pipeline(ignorestatus(cmd),stdout=io,stderr=io));end
    global passed &= success(proc)
    if isfile(report);large_measurements[string(threads)]=TOML.parsefile(report);end
    println(read(log,String));flush(stdout)
end
large_local=joinpath(@__DIR__,"local_large_performance_baseline.toml")
large_baseline=isfile(large_local) ? large_local : joinpath(@__DIR__,"fixtures","large_performance_reference.toml")
if all(haskey(large_measurements,k) && get(large_measurements[k],"passed",false) for k in ("1","4"))
    a=large_measurements["1"];b=large_measurements["4"]
    cross_ok=b["seconds"]<=1.5a["seconds"] && b["allocated_bytes"]<=1.25a["allocated_bytes"]+50_000_000 && b["max_rss_bytes"]<=1.5a["max_rss_bytes"]+128_000_000
    println("Assignment 10x10 cross-thread time/memory: $cross_ok")
    global passed &= cross_ok
    if isfile(large_baseline)
        ref=TOML.parsefile(large_baseline)
        same=get(ref,"hardware",nothing)==hardware && get(ref,"workload",nothing)==a["workload"]
        if !same
            println("Large-workload baseline does not match this hardware/Julia/workload.")
            isfile(large_local) && !calibrate && (global passed=false)
        elseif !calibrate
            for k in ("1","4")
                local now, old, time_ok, allocation_ok, rss_ok
                now=large_measurements[k];old=ref[k]
                time_ok=now["seconds"]<=1.3old["seconds"]+1.
                allocation_ok=now["allocated_bytes"]<=1.2old["allocated_bytes"]+50_000_000
                rss_ok=now["max_rss_bytes"]<=1.25old["max_rss_bytes"]+64_000_000
                println("Assignment 10x10 baseline $k: time=$time_ok, allocations=$allocation_ok, peak RSS=$rss_ok")
                global passed &= time_ok && allocation_ok && rss_ok
            end
        end
    else
        println("No large-workload historical baseline yet; portable and cross-thread checks apply.")
    end
else
    global passed=false
end
@assert sourcehash()==before "Tests modified algorithm source files"

if calibrate
    passed || error("Refusing to bless a performance baseline while correctness tests fail")
    large_measurements["hardware"]=hardware;large_measurements["workload"]=large_measurements["1"]["workload"]
    open(large_local,"w") do io;TOML.print(io,large_measurements);end
    measurements["hardware"]=hardware;measurements["workload"]=workload
    open(localbaseline,"w") do io;TOML.print(io,measurements);end
end
open(joinpath(OUT,"summary.toml"),"w") do io
    TOML.print(io,Dict("passed"=>passed,"wall_seconds"=>time()-started,"hardware"=>hardware,"workload"=>workload,"measurements"=>measurements,"large_measurements"=>large_measurements,"quick_wall_seconds"=>quick_wall_seconds))
end
println("Total wall time: $(round(time()-started;digits=1)) seconds. Overall passed: $passed")
println("Reports: $OUT")
exit(passed ? 0 : 1)
