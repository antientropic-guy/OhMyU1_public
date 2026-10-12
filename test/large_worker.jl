# Isolated process: RSS covers identical imports, warm-up and the large workload
# in both thread layouts, never a different preceding correctness suite.
const STARTED=time()
include("helpers.jl")
include("large_performance.jl")
result=Dict{String,Any}("load_seconds"=>time()-STARTED)
passed=true
try
    @testset "Assignment 10x10 end-to-end performance" begin
        merge!(result,large_performance_tests())
    end
catch err
    global passed=false
    showerror(stderr,err,catch_backtrace());println(stderr)
end
result["passed"]=passed;result["worker_seconds"]=time()-STARTED
open(ARGS[1],"w") do io;TOML.print(io,result);end
exit(passed ? 0 : 1)
