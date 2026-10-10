"""Fixed-size exact-cover family with a variable interaction span.

There are 18 ordered elements, 225 distinct tiles, and b=ones(18).
Tile-size counts (18,62,88,57) are fixed. span=5 contains every allowed
tile; larger spans sample each size without replacement. Adjacent pairs
are mandatory. Columns are ordered by minimum element, with singleton u
AFTER all non-singleton tiles whose minimum is u. No random column shuffle.
This file is included by the experiment scripts, not by the package module.
"""
function locality_instance(rng::AbstractRNG, span::Int; elements=18)
    elements == 18 || error("This protocol fixes 18 elements and 225 variables")
    5 <= span <= elements || error("span must be 5:18")
    catalogue = [Vector{Vector{Int}}() for _ in 1:4]
    function extend!(t)
        push!(catalogue[length(t)],copy(t))
        length(t)==4 && return
        for v in last(t)+1:min(elements,first(t)+span-1)
            push!(t,v); extend!(t); pop!(t)
        end
    end
    for u in 1:elements; extend!([u]); end
    tiles = [[u] for u in 1:elements]
    for (s,needed) in ((2,62),(3,88),(4,57))
        mandatory = s==2 ? [[u,u+1] for u in 1:elements-1] : Vector{Int}[]
        available = filter(t->!(t in mandatory),catalogue[s])
        append!(tiles,mandatory)
        append!(tiles,available[randperm(rng,length(available))[1:needed-length(mandatory)]])
    end
    sort!(tiles;by=t->(first(t),length(t)==1 ? 1 : 0,Tuple(t)))
    A = [Int(u in t) for u in 1:elements,t in tiles]
    @assert size(A)==(18,225) && allunique(Tuple.(tiles))
    (;A,b=ones(Int,elements),tiles,n=15,span,elements,
      column_sizes=length.(tiles),actual_density=sum(A)/length(A))
end

"""Exact full-indicator ranks and number of paths, without building an MPS.

Every reachable nonoverlapping prefix extends to a cover: each uncovered
element still has its singleton available, unless it has just been retired.
Drop a state only when a retired element is uncovered. Thus all retained
prefix charges are extendable, and their count is the exact unfolding rank.
UInt128 path counts are safe: the number of partitions is <= m^m < 2^128.
Independent ExactPartitionSampler counts are checked by the generator.
"""
function locality_ranks(p)
    m,N=size(p.A); @assert m<=18
    masks=[foldl(|,(UInt32(1)<<(u-1) for u in t);init=UInt32(0)) for t in p.tiles]
    full=(UInt32(1)<<m)-1
    # Verify the singleton-last condition on which exactness relies.
    for u in 1:m
        lastpos=findlast(t->u in t,p.tiles)
        @assert p.tiles[lastpos]==[u]
    end
    remaining=fill(UInt32(0),N+1)
    for k in N:-1:1; remaining[k]=remaining[k+1]|masks[k]; end
    ways=zeros(UInt128,1<<m);nextways=similar(ways);fill!(nextways,0)
    ways[1]=1;states=UInt32[0];ranks=Int[1];transitions=Int[]
    for k in 1:N
        next=UInt32[];edges=0;t=masks[k]
        for q in states
            for selected in (false,true)
                selected && q&t!=0 && continue
                v=selected ? q|t : q
                (v|remaining[k+1])==full || continue
                idx=Int(v)+1
                nextways[idx]==0 && push!(next,v)
                nextways[idx]+=ways[Int(q)+1];edges+=1
            end
        end
        for q in states; ways[Int(q)+1]=0; end
        ways,nextways=nextways,ways;states=next
        push!(ranks,length(states));push!(transitions,edges)
    end
    @assert states==[full]
    (;ranks,transitions,total=BigInt(ways[Int(full)+1]),
      maxrank=maximum(ranks[2:end-1]),meanrank=sum(ranks[2:end-1])/(N-1))
end
