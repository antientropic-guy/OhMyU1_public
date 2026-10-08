using OhMyU1, Random
using OhMyU1: OptimizationProblem, assignment_matrix, fill_assignment_vectors!,
    build_mps_from_feasible_samples, ChargeIndex, init_u1_mps

function full_indicator(n)
    A=assignment_matrix(n); b=ones(Int,2n)
    problem=OptimizationProblem(A=A,b=b,cost_function=x->0.0)
    one=zeros(Int,n*n,1)
    fill_assignment_vectors!(one,n)
    template=build_mps_from_feasible_samples(problem,one,1,false)
    links=ChargeIndex{Int}[]
    for k in 0:n*n
        charges=Set{Vector{Int}}()
        if k==n*n
            push!(charges,zeros(Int,2n))
        else
            r,s=divrem(k,n)
            leftmask=(1<<s)-1
            rightmask=((1<<n)-1) ⊻ leftmask
            for mask in 0:(1<<n)-1
                h=count_ones(mask)-r
                h in (0,1) || continue
                h==0 && (rightmask & ~mask)==0 && continue
                h==1 && (leftmask & mask)==0 && continue
                q=ones(Int,2n)
                for j in 1:n
                    q[j]=1-((mask>>(j-1))&1)
                end
                q[n+1:n+r].=0
                q[n+r+1]=1-h
                push!(charges,q)
            end
        end
        push!(links,ChargeIndex(charges))
    end
    init_u1_mps(Float64,links,template.SiteIndices,A,b,1,fill(2,n*n))
end

function main_memory()
root=normpath(joinpath(@__DIR__,"..",".."))
out=joinpath(root,"data","protes_comparison","scaling","memory.csv")
open(out,"w") do io
    println(io,"n,mode,total_bond_max,blocks,numeric_bytes,julia_summarysize_bytes")
    for n in 2:16
        Random.seed!(20261008+n)
        T=zeros(Int,n*n,400);fill_assignment_vectors!(T,n)
        problem=OptimizationProblem(A=assignment_matrix(n),b=ones(Int,2n),cost_function=x->0.0)
        modes=("sample","EELS","full")
        for mode in modes
            mps=mode=="full" ? full_indicator(n) : build_mps_from_feasible_samples(problem,T,1,mode=="EELS")
            count_blocks=sum(length(c.Blocks) for c in mps.Cores)
            numeric=sum(sizeof(block) for core in mps.Cores for block in values(core.Blocks))
            total=Base.summarysize(mps)
            maxrank=maximum(ind.Card for ind in mps.LinkIndices)
            println(io,"$n,$mode,$maxrank,$count_blocks,$numeric,$total")
            println("memory n=$n $mode: $numeric numeric bytes; $total with Julia metadata")
        end
        flush(io);GC.gc()
    end
end
end

abspath(PROGRAM_FILE)==(@__FILE__) && main_memory()
