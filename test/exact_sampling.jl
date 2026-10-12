# Exercise the exact assignment constructor actually used by the benchmark.
module AssignmentIndicatorUnderTest
include(joinpath(@__DIR__,"..","scripts","protes","assignment_memory.jl"))
end
include(joinpath(ROOT,"src","partition_locality.jl"))
# Independent forward/backward subset-sum DP feeds the library constructor.
# There is no public full set-partitioning MPS builder; the production APIs
# under test are init_u1_mps, locality_ranks and ExactPartitionSampler.
function exact_dp_mps(A,b)
    d=size(A,2);zeroq=Tuple(zeros(Int,length(b)))
    forward=[Set([zeroq])]
    for j in 1:d
        push!(forward,Set(Tuple(collect(q)+A[:,j]*x) for q in forward[end] for x in 0:1))
    end
    backward=Vector{Set{typeof(zeroq)}}(undef,d+1);backward[end]=Set([zeroq])
    for j in d:-1:1
        backward[j]=Set(Tuple(collect(q)+A[:,j]*x) for q in backward[j+1] for x in 0:1)
    end
    links=[U.ChargeIndex(Set(collect(q) for q in backward[j] if Tuple(b-collect(q)) in forward[j])) for j in 1:d+1]
    domains=fill([0,1],d)
    # A complete feasible seed is used only to obtain physical-site metadata.
    X=space(d);T=X[:,findall(x->A*x==b,eachcol(X))]
    template=U.build_mps_from_feasible_samples(problem(A,b),T[:,1:1],1,false)
    U.init_u1_mps(Float64,links,template.SiteIndices,A,b,1,fill(2,d))
end
@testset "Exact assignment indicator" begin
    for n in (1,2,3)
        m=AssignmentIndicatorUnderTest.full_indicator(n);X=space(n*n)
        # Independent row/column sums, not assignment_matrix, define feasibility.
        valid=[all(sum(reshape(x,n,n);dims=1).==1)&&all(sum(reshape(x,n,n);dims=2).==1) for x in eachcol(X)]
        @test [m[x] for x in eachcol(X)]==Float64.(valid)
        @test count(valid)==factorial(n)
        U.orthogonalize!(m);@test isapprox(U.u1_norm(m)^2,factorial(n);atol=1e-10)
    end
end
@testset "Exact set partitioning: indicator, ranks, uniform source" begin
    tiles=[[1,2],[1,3],[1],[2,3],[2],[3]]
    A=[Int(u in t) for u in 1:3,t in tiles];b=ones(Int,3);X=space(6)
    valid=[A*x==b for x in eachcol(X)];m=exact_dp_mps(A,b)
    @test [m[x] for x in eachcol(X)]==Float64.(valid)
    @test count(valid)==4
    r=locality_ranks((;A,tiles));s=U.ExactPartitionSampler(tiles,3)
    @test r.total==s.counts[UInt128(0)]==count(valid)
    for k in 1:5
        tensor=reshape(Float64.(valid),2^k,2^(6-k))
        @test rank(tensor)==r.ranks[k+1]==length(m.LinkIndices[k+1].Charges)
    end
    Y=zeros(Int,6,4096);Y2=similar(Y)
    U.fill_partition_vectors!(MersenneTwister(515),Y,s)
    U.fill_partition_vectors!(MersenneTwister(515),Y2,s)
    @test Y==Y2 && all(A*Y .== b)
    @test columns(Y)==columns(X[:,valid])
    for x in eachcol(X[:,valid])
        frequency=count(==(Tuple(x)),Tuple.(eachcol(Y)))/size(Y,2)
        @test abs(frequency-.25)<=8sqrt(.25*.75/size(Y,2))+2/size(Y,2)
    end
    # Exact path probabilities telescope: no frequency-based flaky assertion.
    for x in eachcol(X[:,valid])
        mask=UInt128(0);p=big(1)//big(1)
        while mask!=s.full
            u=trailing_zeros(s.full & ~mask)+1
            j=only([j for j in s.by_cell[u] if x[j]==1 && mask&s.masks[j]==0])
            next=mask|s.masks[j];p*=s.counts[next]//s.counts[mask];mask=next
        end
        @test p==1//r.total
    end
    # Views and concurrent independent RNGs must not modify neighboring columns.
    buffer=fill(-7,6,34)
    U.fill_partition_vectors!(MersenneTwister(3),view(buffer,:,2:33),s)
    @test all(buffer[:,[1,34]].==-7) && all(A*buffer[:,2:33].==b)
    tasks=[Threads.@spawn begin
        Z=zeros(Int,6,32);U.fill_partition_vectors!(MersenneTwister(77),Z,s);Z
    end for _ in 1:4]
    zs=fetch.(tasks);@test all(==(first(zs)),zs)
end
