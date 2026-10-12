#!/usr/bin/env julia
include("run_PartitionLocality.jl")

# Independently enumerate every cover of a tiny problem and explicitly form
# the nonzero rows/columns of each unfolding. This checks the rank theorem
# and the frontier implementation, not merely the same DP twice.
function check_small_locality()
    m=6
    tiles=[[u] for u in 1:m]
    append!(tiles,[[u,v] for u in 1:m for v in u+1:m])
    append!(tiles,[[1,3,6],[2,4,5],[1,2,3,4]])
    sort!(tiles;by=t->(first(t),length(t)==1 ? 1 : 0,Tuple(t)))
    A=[Int(u in t) for u in 1:m,t in tiles]
    p=(;A,tiles);computed=locality_ranks(p)
    s=ExactPartitionSampler(tiles,m)
    covers=Vector{Int}[]
    function visit(mask,x)
        if mask==s.full;push!(covers,copy(x));return;end
        u=trailing_zeros(s.full & ~mask)+1
        for j in s.by_cell[u]
            mask&s.masks[j]==0 || continue
            x[j]=1;visit(mask|s.masks[j],x);x[j]=0
        end
    end
    visit(UInt128(0),zeros(Int,length(tiles)))
    @assert computed.total==length(covers)==s.counts[UInt128(0)]
    # Verify every cover probability exactly, with rational arithmetic.
    for x in covers
        mask=UInt128(0);prob=big(1)//big(1)
        while mask!=s.full
            u=trailing_zeros(s.full & ~mask)+1
            candidates=[j for j in s.by_cell[u] if mask&s.masks[j]==0 && x[j]==1]
            @assert length(candidates)==1
            next=mask|s.masks[only(candidates)]
            prob*=s.counts[next]//s.counts[mask];mask=next
        end
        @assert prob==1//computed.total
    end
    for k in 1:length(tiles)-1
        left=unique(Tuple(x[1:k]) for x in covers)
        right=unique(Tuple(x[k+1:end]) for x in covers)
        li=Dict(x=>i for (i,x) in enumerate(left));ri=Dict(x=>i for (i,x) in enumerate(right))
        unfolding=zeros(length(left),length(right))
        for x in covers;unfolding[li[Tuple(x[1:k])],ri[Tuple(x[k+1:end])]]=1;end
        @assert rank(unfolding)==computed.ranks[k+1]
    end
    X=zeros(Int,length(tiles),200);Y=similar(X)
    fill_partition_vectors!(MersenneTwister(1234),X,s)
    fill_partition_vectors!(MersenneTwister(1234),Y,s)
    @assert X==Y && all(A*X .== 1)
    println("Independent dense-unfolding validation passed: $(length(covers)) exact covers")
end

function audit_locality_inputs()
    signatures=String[]
    for span in LOCALITY_SPANS,i in 1:40
        d=load(locality_path(span,i));p=d["instance"]
        @assert size(p.A)==(18,225) && all(p.b.==1)
        @assert [count(==(j),p.column_sizes) for j in 1:4]==[18,62,88,57]
        @assert all(maximum(t)-minimum(t)+1<=span for t in p.tiles)
        @assert all(p.A*Int.(d["initial"]) .== p.b)
        @assert allunique(Tuple.(p.tiles)) && length(d["structural"].ranks)==226
        s=ExactPartitionSampler(p.tiles,18)
        @assert s.counts[UInt128(0)]==d["structural"].total
        @assert locality_ranks(p).ranks==d["structural"].ranks
        @assert locality_instance(MersenneTwister(d["specification"].seed),span).A==p.A
        X=zeros(Int,225,500)
        fill_partition_vectors!(MersenneTwister(d["specification"].seed+100_000_000),X,s)
        @assert X==d["initial"]
        push!(signatures,bytes2hex(sha256(read(locality_path(span,i)))))
    end
    @assert length(signatures)==280
    println("280 inputs verified: fixed sizes, feasibility, DP counts, initial RNG replay")
end
function check_locality_revision()
    check_small_locality()
    audit_locality_inputs()
    for span in (5,18)
        d=load(locality_path(span,1));p=d["instance"];X=Int.(d["initial"])
        problem=OptimizationProblem(A=p.A,b=p.b,cost_function=x->0.,name="index audit")
        mps=OhMyU1.build_mps_from_feasible_samples(problem,X,1,false)
        charges=OhMyU1.compute_link_charges(p.A,p.b,X)
        bad=0;total=0
        for j in 1:size(X,1)-1
            L=OhMyU1.collect_left(mps.Cores[j]);R=OhMyU1.collect_right(mps.Cores[j+1])
            for k in axes(X,2)
                q=charges[j+1,:,k]
                row=OhMyU1.nondeg_merged_index(L,q,mps.Cores[j].Indices[2],X[j,k])
                col=OhMyU1.nondeg_merged_index(R,q,mps.Cores[j+1].Indices[2],X[j+1,k])
                bad+=(row!=1 || col!=1);total+=1
            end
        end
        println("Original data span=$span instance=1: incorrectly addressed sample/bond terms $bad / $total")
    end
    for objective in ("linear","entropic"),grid in (false,true),span in (5,18)
        run_locality(span;objective,grid,smoke=true,elements=grid ? 20 : 18,n=grid ? 13 : 15,instances=1:1)
    end
    println("All eight paired smoke settings passed (32 solver arms).")
end

if abspath(PROGRAM_FILE)==@__FILE__
    check_locality_revision()
end
