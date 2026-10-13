# Exact indicator construction and feasible SA proposals.
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

# Each fixed patch has a precomputed list of all local exact covers. A proposal
# is uniform on this list if the current cover has no tile crossing its boundary;
# otherwise it is a self-loop. Forward/reverse probabilities are identical.
function patch_catalogue(p)
    masks=[foldl(|,(UInt64(1)<<(i-1) for i in t);init=UInt64(0)) for t in p.tiles]
    patches=Any[]
    for patch in masks
        inside=findall(t->t&patch==t,masks)
        covers=Vector{Int}[]
        function visit(covered,chosen)
            covered==patch && (push!(covers,copy(chosen));return)
            bit=UInt64(1)<<trailing_zeros(patch & ~covered)
            for j in inside
                masks[j]&bit==0 && continue
                masks[j]&covered==0 || continue
                push!(chosen,j);visit(covered|masks[j],chosen);pop!(chosen)
            end
        end
        visit(UInt64(0),Int[])
        push!(patches,(patch,inside,covers))
    end
    masks,patches
end
function patch_neighbor!(x,y,rng,masks,patches)
    copyto!(y,x)
    patch,inside,covers=rand(rng,patches)
    for j in eachindex(x)
        x[j]==0 && continue
        overlap=masks[j]&patch
        overlap!=0 && overlap!=masks[j] && return nothing
    end
    for j in inside;y[j]=0;end
    for j in rand(rng,covers);y[j]=1;end
    nothing
end

# Enumerate reachable prefix coverage masks and prune states whose uncovered
# cells no longer occur in a suffix. Singleton-last order guarantees completion.
function partition_indicator(p,initial,total)
    A=p.A;m,N=size(A);allmask=(UInt64(1)<<m)-1
    masks=[foldl(|,(UInt64(1)<<(i-1) for i in t);init=UInt64(0)) for t in p.tiles]
    suffix=zeros(UInt64,N+1)
    for j in N:-1:1;suffix[j]=suffix[j+1]|masks[j];end
    states=Set(UInt64[0]);links=ChargeIndex{Int}[]
    charge(q)=[Int((q>>(i-1))&1==0) for i in 1:m]
    push!(links,ChargeIndex(Set([charge(UInt64(0))])))
    for j in 1:N
        next=Set{UInt64}()
        for q in states, x in (false,true)
            x && q&masks[j]!=0 && continue
            v=x ? q|masks[j] : q
            ((allmask & ~v) & ~suffix[j+1])==0 && push!(next,v)
        end
        states=next;push!(links,ChargeIndex(Set(charge(q) for q in states)))
    end
    problem=OptimizationProblem(A=A,b=p.b,cost_function=x->0.)
    template=build_mps_from_feasible_samples(problem,initial[:,1:1],1,false)
    mps=init_u1_mps(Float64,links,template.SiteIndices,A,p.b,1,fill(2,N))
    counts=Dict(q=>1. for q in mps.LinkIndices[end].Charges)
    for core in reverse(mps.Cores)
        previous=Dict(q=>0. for q in core.Indices[1].Charges)
        for ((l,s,r),block) in core.Blocks;previous[l]+=counts[r];end
        for ((l,s,r),block) in core.Blocks;block[1]=sqrt(counts[r]/previous[l]);end
        counts=previous
    end
    @assert isapprox(only(values(counts)),Float64(total);rtol=1e-12)
    mps.ort_center=1
    @assert isapprox(u1_norm(mps),1;atol=1e-12)
    mps
end

function validate_patches(p,initial,masks,patches)
    rng=MersenneTwister(19);x=Float64.(initial[:,1]);y=similar(x)
    for k in 1:1000
        patch_neighbor!(x,y,rng,masks,patches)
        @assert all(p.A*y .== 1)
        x,y=y,x
    end
    for (patch,inside,covers) in patches
        @assert allunique(Tuple.(covers))
        for cover in covers
            @assert sum(count_ones(masks[j]) for j in cover)==count_ones(patch)
            @assert foldl(|,(masks[j] for j in cover);init=UInt64(0))==patch
        end
    end
end
