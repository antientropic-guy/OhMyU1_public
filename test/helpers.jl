using Test, Random, LinearAlgebra, Statistics, TOML, SHA
using OhMyU1
const U = OhMyU1
const ROOT = normpath(joinpath(@__DIR__,".."))
BLAS.set_num_threads(1)
bits(k,d) = Int[(k>>(j-1))&1 for j in 1:d]
space(d) = reduce(hcat,(bits(k,d) for k in 0:2^d-1))
columns(X) = Set(Tuple.(eachcol(X)))
problem(A,b;cost=x->sum(x)) = U.OptimizationProblem(A=A,b=b,cost_function=cost,name="test")
function fixtures()
    [begin
        A=reduce(vcat,permutedims.(f["A"]));b=Int.(f["b"])
        T=reduce(hcat,[bits(k,size(A,2)) for k in f["training_ids"]])
        (;name=f["name"],A,b,T)
    end for f in TOML.parsefile(joinpath(@__DIR__,"fixtures","problems.toml"))["problem"]]
end
build(f;T=f.T) = U.build_mps_from_feasible_samples(problem(f.A,f.b),copy(T),1,false)
function observed_charges(A,b,T)
    [Set(Tuple(b-A[:,1:k]*x[1:k]) for x in eachcol(T)) for k in 0:size(A,2)]
end
function closure(f)
    Q=observed_charges(f.A,f.b,f.T);X=space(size(f.A,2))
    [f.A*x==f.b && all(Tuple(f.b-f.A[:,1:k]*x[1:k]) in Q[k+1] for k in 0:length(x)) for x in eachcol(X)]
end
# Dense tensors are built directly from block keys and physical charges.
# No library getindex, getval, collect/split, or merged-index helper is used.
function dense_cores(m)
    qs=[sort!(collect(q.Charges);by=Tuple) for q in m.LinkIndices]
    cores=Array{Float64,3}[]
    for (j,c) in enumerate(m.Cores)
        G=zeros(length(qs[j]),length(c.Indices[2].FlatDomain),length(qs[j+1]))
        for ((l,s,r),B) in c.Blocks
            a=findfirst(==(l),qs[j]);b=findfirst(==(r),qs[j+1])
            xs=findall(x->c.Indices[2].A_vec*x==s,c.Indices[2].FlatDomain)
            @assert size(B)==(1,length(xs),1)
            for (h,x) in enumerate(xs);G[a,x,b]=B[1,h,1];end
        end
        push!(cores,G)
    end
    cores,qs
end
function amplitude(G,x)
    v=ones(1)
    for j in eachindex(G);v=vec(transpose(v)*G[j][:,x[j]+1,:]);end
    only(v)
end
amplitudes(m,X) = begin
    G,_=dense_cores(m);[amplitude(G,x) for x in eachcol(X)]
end
function dense_loss(a,ids,p)
    sum(p)*log(sum(abs2,a))-sum(p[j]*log(abs2(a[ids[j]])) for j in eachindex(p) if p[j]!=0)
end
sampleids(X)=[1+sum(x[j]<<(j-1) for j in eachindex(x)) for x in eachcol(X)]
function randomized!(m;seed=82)
    rng=MersenneTwister(seed)
    for c in m.Cores,B in values(c.Blocks); B .= (.4 .+ rand(rng,size(B)...)).*rand(rng,[-1.,1.],size(B)...);end
    m.ort_center=0;U.orthogonalize!(m);U.normalize!(m);m
end
function assert_canonical(m)
    center=m.ort_center
    @test center in eachindex(m.Cores)
    for j in eachindex(m.Cores)
        j==center && continue
        blocks=j<center ? U.collect_left(m.Cores[j]).Blocks : U.collect_right(m.Cores[j]).Blocks
        for B in values(blocks)
            gram=j<center ? B'*B : B*B'
            @test isapprox(gram,Matrix{Float64}(I,size(gram)...);atol=2e-11,rtol=2e-11)
        end
    end
end
