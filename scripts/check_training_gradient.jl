#!/usr/bin/env julia
using OhMyU1, LinearAlgebra, Random, Test
const U=OhMyU1
BLAS.set_num_threads(1)
function audit(A,b)
    d=size(A,2)
    X=reduce(hcat,[Int[(k>>(j-1))&1 for j in 1:d] for k in 0:2^d-1 if A*Int[(k>>(j-1))&1 for j in 1:d]==b])
    p=U.OptimizationProblem(A=A,b=b,cost_function=x->sum(x),name="gradient audit")
    m=U.build_mps_from_feasible_samples(p,X,1,false)
    rng=MersenneTwister(81)
    for c in m.Cores,B in values(c.Blocks);B .= .5 .+ rand(rng,size(B)...);end
    U.orthogonalize!(m);U.normalize!(m)
    weights=collect(1.:size(X,2));weights./=sum(weights)
    maxerr=0.;olderr=0.
    for direction in (:right,:left)
        for i in (direction==:right ? (1:d-1) : (d-1:-1:1))
            merged=U.MergedCores(m.Cores[i],m.Cores[i+1]);L=merged.left_matrix;R=merged.right_matrix
            charges=U.compute_link_charges(A,b,X)
            amplitudes=[m[x] for x in eachcol(X)]
            locations=[]
            # Independently locate sectors using physical charges and slices.
            for (j,x) in enumerate(eachcol(X))
                q=charges[i+1,:,j]
                lc=A[:,i]*x[i];rc=A[:,i+1]*x[i+1]
                li=findfirst(==(lc),L.DegeneracyDict[q]);ri=findfirst(==(rc),R.DegeneracyDict[q])
                row=L.XboundsDict[q][li]+m.Cores[i].Indices[2].InvXindex[x[i]][2]
                col=R.XboundsDict[q][ri]+m.Cores[i+1].Indices[2].InvXindex[x[i+1]][2]
                push!(locations,(q,row,col))
            end
            Z=sum(abs2,amplitudes)
            @test isapprox(Z,U.u1_norm(m)^2)
            @test isapprox(Z,sum(sum(abs2,B) for B in values(merged.Blocks)))
            for (q,B) in merged.Blocks,idx in CartesianIndices(B)
                g=2B[idx]/Z;gold=g
                for (j,x) in enumerate(eachcol(X))
                    qq,row,col=locations[j]
                    if qq==q
                        denom=U.getval(m.Cores[i],charges[i,:,j],x[i])*U.getval(m.Cores[i+1],x[i+1],charges[i+2,:,j])
                        row2=U.nondeg_merged_index(L,q,m.Cores[i].Indices[2],x[i])
                        col2=U.nondeg_merged_index(R,q,m.Cores[i+1].Indices[2],x[i+1])
                        @test (row,col)==(row2,col2)
                        idx==CartesianIndex(row2,col2) && (g-=2weights[j]/denom)
                        # Old nonzero-column behavior; zero columns previously threw.
                        idx==CartesianIndex(1,1) && (gold-=2weights[j]/denom)
                    end
                end
                function loss(delta)
                    amps=copy(amplitudes)
                    for j in eachindex(amps)
                        qq,row,col=locations[j]
                        qq==q && idx==CartesianIndex(row,col) && (amps[j]*=(B[idx]+delta)/B[idx])
                    end
                    log(sum(abs2,amps))-sum(weights.*log.(abs2.(amps)))
                end
                fd=(loss(1e-6)-loss(-1e-6))/2e-6
                maxerr=max(maxerr,abs(fd-g));olderr=max(olderr,abs(fd-gold))
                @test isapprox(g,fd;atol=2e-7,rtol=2e-6)
            end
            # Move the center exactly without changing amplitudes.
            for (q,B) in merged.Blocks
                F=svd(B)
                if direction==:right
                    L.Blocks[q]=F.U[:,1:1];R.Blocks[q]=F.S[1]*F.Vt[1:1,:]
                else
                    L.Blocks[q]=F.U[:,1:1]*F.S[1];R.Blocks[q]=F.Vt[1:1,:]
                end
            end
            m.Cores[i]=U.split_left(L);m.Cores[i+1]=U.split_right(R)
            m.ort_center=direction==:right ? i+1 : i
        end
    end
    before=U.Loss(m,X,weights)
    U.train_nondeg!(m,1,X,weights,U.TrainParams(1e-4,1))
    after=U.Loss(m,X,weights)
    @test after<before
    @test isapprox(after,log(sum(m[x]^2 for x in eachcol(X)))-sum(weights[j]*log(m[x]^2) for (j,x) in enumerate(eachcol(X))))
    samples=zeros(Int,d,200);U.normalize!(m)
    # Check exact serial-sampler conditionals against the Born probabilities.
    for x in eachcol(X)
        q=copy(b);prob=1.
        for j in 1:d
            c=m.Cores[j];domain=c.Indices[2].FlatDomain
            z=sum(U.getval(c,q,v)^2 for v in domain)
            prob*=U.getval(c,q,x[j])^2/z
            q .-= A[:,j]*x[j]
        end
        @test isapprox(prob,m[x]^2)
    end
    Random.seed!(781);U.sample_nondeg!(m,samples,200)
    replay=similar(samples);Random.seed!(781);U.sample_nondeg!(m,replay,200)
    @test replay==samples
    @test all(A*samples .== b)
    println("A=$(repr(A)): max FD error=$maxerr; old-index error=$olderr; loss $before -> $after")
end
@testset "Merged-core normalized cross entropy" begin
    audit(ones(Int,1,5),[2])
    audit([1 0 1 0 1;0 1 0 1 1],[1,1])
    audit([1 0 1 0],[1])
end
