# Performance runs first in both workers: peak RSS then covers comparable
# imports + the same benchmark, not different preceding correctness suites.
const STARTED=time()
include("helpers.jl")
const RESULTS=Dict{String,Any}("load_seconds"=>time()-STARTED)
const STAGES=Dict{String,Float64}()
failed=false
include("performance.jl")
stage_started=time()
try
    @testset "Performance" begin
        merge!(RESULTS,performance_tests())
    end
catch err
    global failed=true
    showerror(stderr,err,catch_backtrace());println(stderr)
end
STAGES["performance_including_jit"]=time()-stage_started
files="--threaded" in ARGS ? ("sampling.jl","selection.jl","fast_paths.jl") : ("structure.jl","index_lookup.jl","exact_sampling.jl","sampling.jl","gradients.jl","selection.jl","fast_paths.jl")
for file in files
    file_started=time()
    try
        @testset "$file" begin
            include(file)
        end
    catch err
        global failed=true
        showerror(stderr,err,catch_backtrace());println(stderr)
    end
    STAGES[file]=time()-file_started
end
RESULTS["stage_seconds"]=STAGES
RESULTS["passed"]=!failed;RESULTS["worker_seconds"]=time()-STARTED
open(ARGS[1],"w") do io;TOML.print(io,RESULTS);end
exit(failed ? 1 : 0)
