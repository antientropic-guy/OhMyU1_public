include(joinpath(@__DIR__, "assignment_memory.jl"))
using LinearAlgebra, DelimitedFiles
using OhMyU1: sample_nondeg!, u1_norm, getval
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("Use one Julia thread")

# Exact backward dynamic program. Every raw block is one, so suffix counts
# give a right-canonical normalized amplitude sqrt(1/n!) on every permutation.
# This avoids large temporary dictionaries of the generic LQ canonicalizer.
# Sampling itself is the unchanged library sample_nondeg! implementation.
function canonical_full!(mps,n)
    suffix=Dict(q=>1.0 for q in mps.LinkIndices[end].Charges)
    for core in reverse(mps.Cores)
        counts=Dict(q=>0.0 for q in core.Indices[1].Charges)
        for ((left,site,right),block) in core.Blocks
            counts[left]+=suffix[right]
        end
        for ((left,site,right),block) in core.Blocks
            block[1]=sqrt(suffix[right]/counts[left])
        end
        suffix=counts
    end
    total=only(values(suffix))
    isapprox(total,Float64(factorial(big(n)));rtol=1e-12) || error("Wrong support count")
    mps.ort_center=1
    isapprox(u1_norm(mps),1;atol=1e-12) || error("Wrong normalization")
    total
end

function benchmark_full_u1()
    root=normpath(joinpath(@__DIR__,"..",".."))
    out=joinpath(root,"data","protes_comparison","scaling","full_u1.csv")
    minimum_n=isempty(ARGS) ? 4 : parse(Int,ARGS[1])
    cached_sizes=Dict{Int,Int}()
    if length(ARGS)>=2
        cached=readdlm(ARGS[2],',',Any;skipstart=1)
        for row in eachrow(cached)
            cached_sizes[Int(row[1])]=Int(row[14])
        end
    end
    open(out,minimum_n==4 ? "w" : "a") do io
        minimum_n==4 && println(io,"n,repetition,batch,batches,samples,seconds,samples_per_second,unique,novel_unique,feasible,build_seconds,canonical_seconds,numeric_bytes,object_bytes,support")
        for n in minimum_n:16
            GC.gc()
            t=time_ns(); mps=full_indicator(n); build=(time_ns()-t)/1e9
            println("Built complete U1 n=$n in $build s; measuring object memory");flush(stdout)
            numeric=sum(sizeof(b) for c in mps.Cores for b in values(c.Blocks))
            object_bytes=haskey(cached_sizes,n) ? cached_sizes[n] : Base.summarysize(mps)
            t=time_ns();support=canonical_full!(mps,n);canonical=(time_ns()-t)/1e9
            GC.gc() # release size-traversal and canonicalization temporary objects before timing
            samples=zeros(Int,n*n,100)
            sample_nondeg!(mps,samples,100) # compilation and warmup excluded
            input=joinpath(root,"data","protes_comparison","scaling","paired_inputs","linear_$(n)_0.i32")
            initial=Int.(reshape(reinterpret(Int32,read(input)),n*n,400))
            initial_set=Set(Tuple(x) for x in eachcol(initial))
            for rep in 0:2
                Random.seed!(20261008+n*1000+rep)
                elapsed=0.0;batches=0;seen=Set{Tuple}();feasible=0
                while elapsed<1.0
                    t=time_ns(); sample_nondeg!(mps,samples,100); elapsed+=(time_ns()-t)/1e9
                    # Diagnostic bookkeeping excluded symmetrically in both samplers.
                    all(mps.A*samples .== 1) || error("Infeasible sample")
                    feasible+=100;union!(seen,(Tuple(x) for x in eachcol(samples)));batches+=1
                end
                novel=length(setdiff(seen,initial_set))
                println(io,join((n,rep,100,batches,100batches,elapsed,100batches/elapsed,length(seen),novel,feasible,build,canonical,numeric,object_bytes,support),','))
                println("Full U1 n=$n rep=$rep: $(round(100batches/elapsed)) samples/s, $(length(seen)) unique, object=$object_bytes")
                flush(io);flush(stdout)
            end
            mps=nothing;GC.gc()
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && benchmark_full_u1()
