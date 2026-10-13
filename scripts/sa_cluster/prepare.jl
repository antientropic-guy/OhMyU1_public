# Build a self-contained input snapshot without running optimizers.
include("adapters.jl")

function assignment_profile(n)
    ranks=Int[]
    for k in 0:n*n
        if k==n*n;push!(ranks,1);continue;end
        r,s=divrem(k,n);left=(1<<s)-1;right=((1<<n)-1) ⊻ left
        count=0
        for mask in 0:(1<<n)-1
            h=count_ones(mask)-r
            h in (0,1) || continue
            h==0 && (right & ~mask)==0 && continue
            h==1 && (left & mask)==0 && continue
            count+=1
        end
        push!(ranks,count)
    end
    ranks
end

function partition_profile(p)
    m,N=size(p.A);allmask=(UInt64(1)<<m)-1
    masks=[foldl(|,(UInt64(1)<<(u-1) for u in t);init=UInt64(0)) for t in p.tiles]
    suffix=zeros(UInt64,N+1)
    for j in N:-1:1;suffix[j]=suffix[j+1]|masks[j];end
    counts=Dict(UInt64(0)=>big(1));ranks=[1]
    for j in 1:N
        next=Dict{UInt64,BigInt}()
        for (q,ways) in counts,chosen in (false,true)
            chosen && q&masks[j]!=0 && continue
            v=chosen ? q|masks[j] : q
            ((allmask & ~v) & ~suffix[j+1])==0 || continue
            next[v]=get(next,v,big(0))+ways
        end
        counts=next;push!(ranks,length(counts))
    end
    @assert length(counts)==1 && haskey(counts,allmask)
    ranks,counts[allmask]
end

function atomic_save(path;kwargs...)
    temporary=path*".pending"
    jldsave(temporary,true,IOStream;kwargs...)
    mv(temporary,path;force=true)
end

function prepare(;out=joinpath(ROOT,"data","sa_cluster_v1"),sizes=(12,14),count=40)
    isfile(joinpath(out,"manifest.jld2")) && error("Input snapshot already exists; do not overwrite a running experiment")
    mkpath(joinpath(out,"inputs"))
    records=NamedTuple[]
    for n in sizes
        aranks=assignment_profile(n)
        for rep in 1:count
            source=joinpath(ROOT,"data","set_partitioning","test","instances","light","instance_$(n)_$(rep)_test.jld2")
            qsource=joinpath(ROOT,"data","test_random_bilinear_forms","q_$(n)_$(rep)_r=7_test.jld2")
            d=load(source);raw=d["instance"];old=d["objectives"]
            order=sortperm(raw.tiles;by=t->(first(t),length(t)==1 ? 1 : 0,Tuple(t)))
            tiles=raw.tiles[order];m=length(raw.b)
            p=(A=[Int(u in t) for u in 1:m,t in tiles],b=ones(Int,m),tiles=tiles,n=n)
            sampler=ExactPartitionSampler(tiles,m)
            sranks,total=partition_profile(p)
            @assert total==sampler.counts[UInt128(0)]==d["exact_cover_count"]
            Q=Matrix{Float64}(only(v for v in values(load(qsource)) if v isa AbstractMatrix && size(v)==(n*n,n*n)))
            # Identical cost arrays and scaling for both feasible sets at each (n, rep).
            entropic=(c=Float64.(old.c[order]),scenarios=Matrix{Float64}(old.scenarios[:,order]),
                      tau=Float64(old.tau),scale=sqrt(Float64(m)))
            @assert size(entropic.scenarios)==(32,n*n) && entropic.tau>0
            for family in ("assignment","partition")
                seed=Int(d["specification"].seed)
                initial=zeros(Int,n*n,400)
                rng=MersenneTwister(seed+100_000_000)
                if family=="partition"
                    fill_partition_vectors!(rng,initial,sampler)
                    A=p.A;b=p.b;ranks=sranks;support=total
                    @assert all(raw.A*initial[invperm(order),:] .== raw.b)
                else
                    for x in eachcol(initial)
                        for (i,j) in enumerate(randperm(rng,n));x[(i-1)*n+j]=1;end
                    end
                    A=assignment_matrix(n);b=ones(Int,2n);ranks=aranks;support=factorial(big(n))
                end
                @assert all(A*initial .== b) && all(x->x in (0,1),initial)
                record=(version=1,family=family,n=n,repetition=rep,seed=seed,A=A,b=b,initial=initial,Q=Q,
                    entropic=entropic,tiles=family=="partition" ? tiles : Vector{Int}[],
                    variable_order=family=="partition" ? order : collect(1:n*n),
                    ranks=ranks,maxrank=maximum(ranks[2:end-1]),meanrank=mean(ranks[2:end-1]),support=string(support),
                    initial_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(Int64.(initial))))),
                    q_sha256=bytes2hex(sha256(reinterpret(UInt8,vec(Q)))),
                    source_partition=relpath(source,ROOT),source_partition_sha256=bytes2hex(sha256(read(source))),
                    source_quadratic=relpath(qsource,ROOT),source_quadratic_sha256=bytes2hex(sha256(read(qsource))))
                name="$(family)_n$(n)_i$(rep).jld2"
                path=joinpath(out,"inputs",name)
                atomic_save(path;record)
                push!(records,(family=family,n=n,instance=rep,file=name,sha256=bytes2hex(sha256(read(path))),
                    variables=n*n,equalities=length(b),maxrank=record.maxrank,meanrank=record.meanrank,support=record.support))
            end
            println("Prepared paired n=$n instance=$rep: assignment rank=$(maximum(aranks)), partition rank=$(maximum(sranks))");flush(stdout)
        end
    end
    manifest=(version=1,count=count,sizes=collect(sizes),initial_count=400,objectives=["quadratic","entropic"],
              budgets=[10,50,200],records=records)
    atomic_save(joinpath(out,"manifest.jld2");manifest)
    open(joinpath(out,"rank_summary.csv"),"w") do io
        println(io,"family,n,instances,mean_maxrank,sd_maxrank,mean_meanrank,sd_meanrank,min_maxrank,max_maxrank")
        for family in ("assignment","partition"),n in sizes
            g=filter(r->r.family==family && r.n==n,records)
            println(io,join((family,n,length(g),mean(r.maxrank for r in g),std(r.maxrank for r in g),
                mean(r.meanrank for r in g),std(r.meanrank for r in g),minimum(r.maxrank for r in g),maximum(r.maxrank for r in g)),','))
        end
    end
end

abspath(PROGRAM_FILE)==(@__FILE__) && prepare()
