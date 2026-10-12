# Exact-cover locality family. N=n^2 binary variables; elements equalities.
# All singletons and adjacent pairs are mandatory. Other distinct tiles of
# sizes 2:4 are sampled uniformly within size; counts use largest remainders
# of the original 62:88:57 proportions. Singleton-last ordering is essential.
function locality_instance(rng::AbstractRNG, span::Int; elements=18, n=15)
    N=n^2
    raw=(N-elements).*[62,88,57]./207
    counts=floor.(Int,raw)
    for j in sortperm(raw.-counts;rev=true)[1:N-elements-sum(counts)]; counts[j]+=1; end
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
    for (s,needed) in zip(2:4,counts)
        mandatory = s==2 ? [[u,u+1] for u in 1:elements-1] : Vector{Int}[]
        available = filter(t->!(t in mandatory),catalogue[s])
        @assert length(mandatory)<=needed<=length(catalogue[s]) "Insufficient distinct tiles"
        append!(tiles,mandatory)
        append!(tiles,available[randperm(rng,length(available))[1:needed-length(mandatory)]])
    end
    sort!(tiles;by=t->(first(t),length(t)==1 ? 1 : 0,Tuple(t)))
    A = [Int(u in t) for u in 1:elements,t in tiles]
    @assert size(A)==(elements,N) && allunique(Tuple.(tiles))
    (;A,b=ones(Int,elements),tiles,n,span,elements,
      column_sizes=length.(tiles),actual_density=sum(A)/length(A))
end

# Exact full-indicator unfolding ranks via a sliding coverage frontier.
# Every retained prefix has a singleton completion. Different charges have
# disjoint suffix supports, hence each nonempty charge block has rank one.
function locality_ranks(p)
    span=maximum(maximum(t)-minimum(t)+1 for t in p.tiles)
    m,N=size(p.A)
    # Every solution is a set partition; Bell(m) bounds every prefix path count
    # because all retained prefixes have an injective singleton completion.
    bell=BigInt[1]
    for n in 1:m;push!(bell,sum(binomial(big(n-1),k)*bell[k+1] for k in 0:n-1));end
    @assert last(bell)<typemax(UInt128)
    width=min(span,m)
    ways=zeros(UInt128,1<<width);other=similar(ways);fill!(other,0)
    ways[1]=1;states=UInt32[0];ranks=Int[1];transitions=Int[];base=1
    for tile in p.tiles
        @assert minimum(tile)==base && maximum(tile)-base<width
        singleton=length(tile)==1
        t=foldl(|,(UInt32(1)<<(u-base) for u in tile);init=UInt32(0))
        next=UInt32[];edges=0
        for q in states
            for chosen in (false,true)
                chosen && (q&t)!=0 && continue
                v=chosen ? q|t : q
                if singleton
                    v&UInt32(1)==0 && continue
                    v>>=1
                end
                idx=Int(v)+1
                other[idx]==0 && push!(next,v)
                other[idx]+=ways[Int(q)+1];edges+=1
            end
        end
        for q in states;ways[Int(q)+1]=0;end
        ways,other=other,ways;states=next
        singleton && (base+=1)
        push!(ranks,length(states));push!(transitions,edges)
    end
    @assert base==m+1 && states==UInt32[0]
    (;ranks,transitions,total=BigInt(ways[1]),maxrank=maximum(ranks[2:end-1]),meanrank=sum(ranks[2:end-1])/(N-1))
end
