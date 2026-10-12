# Dense external environments give the derivative with respect to a full
# two-site tensor. Build the Jacobian independently of getval and train.jl.
function merged_reference(m,i,T,p)
    G,qs=dense_cores(m);X=space(length(G));ids=sampleids(T)
    merged=U.MergedCores(m.Cores[i],m.Cores[i+1]);L=merged.left_matrix;R=merged.right_matrix
    coords=[];v=Float64[];features=Vector{Float64}[]
    for q in sort!(collect(keys(merged.Blocks));by=Tuple)
        rows=[];cols=[]
        for s in L.DegeneracyDict[q]
            left=findfirst(==(q+s),qs[i])
            for x in 0:1;m.A[:,i]*x==s && push!(rows,(left,x));end
        end
        for s in R.DegeneracyDict[q]
            right=findfirst(==(q-s),qs[i+2])
            for x in 0:1;m.A[:,i+1]*x==s && push!(cols,(right,x));end
        end
        for idx in CartesianIndices(merged.Blocks[q])
            a,x=rows[idx[1]];b,y=cols[idx[2]]
            feature=zeros(size(X,2))
            for (h,z) in enumerate(eachcol(X))
                (z[i]==x && z[i+1]==y) || continue
                left=ones(1);right=ones(1)
                for k in 1:i-1;left=vec(transpose(left)*G[k][:,z[k]+1,:]);end
                for k in length(G):-1:i+2;right=G[k][:,z[k]+1,:]*right;end
                feature[h]=left[a]*right[b]
            end
            push!(coords,(q,idx));push!(v,merged.Blocks[q][idx]);push!(features,feature)
        end
    end
    J=reduce(hcat,features);amps=J*v
    @test isapprox(amps,[amplitude(G,x) for x in eachcol(X)];atol=2e-12)
    f(z)=dense_loss(J*z,ids,p)
    fd=similar(v)
    for k in eachindex(v)
        vp=copy(v);vm=copy(v);vp[k]+=1e-6;vm[k]-=1e-6
        fd[k]=(f(vp)-f(vm))/2e-6
    end
    analytic=2sum(p)*(J'*amps)/sum(abs2,amps)-2J[ids,:]'*(p./amps[ids])
    @test all(isapprox.(fd,analytic;atol=3e-7,rtol=3e-6))
    # Compare the actual nondegenerate formula at every merged component.
    actual=2sum(p).*v./U.u1_norm(m)^2
    for (t,x) in enumerate(eachcol(T))
        left=m.A[:,1:i-1]*x[1:i-1];q=first(m.LinkIndices[1].Charges)-left-m.A[:,i]*x[i]
        row=U.nondeg_merged_index(L,q,m.Cores[i].Indices[2],x[i])
        col=U.nondeg_merged_index(R,q,m.Cores[i+1].Indices[2],x[i+1])
        k=findfirst(==((q,CartesianIndex(row,col))),coords)
        actual[k]-=2p[t]/v[k]
    end
    @test all(isapprox.(actual,fd;atol=3e-7,rtol=3e-6))
    merged,coords,fd
end
function reference_sweep!(m,T,p,alpha,direction)
    for i in (direction==:right ? (1:length(m.Cores)-1) : (length(m.Cores)-1:-1:1))
        merged,coords,fd=merged_reference(m,i,T,p)
        for (k,(q,idx)) in enumerate(coords);merged.Blocks[q][idx]-=alpha*fd[k];end
        L=merged.left_matrix;R=merged.right_matrix
        for (q,B) in merged.Blocks
            F=svd(B)
            L.Blocks[q]=F.U[:,1:1]*(direction==:left ? F.S[1] : 1.)
            R.Blocks[q]=F.Vt[1:1,:]*(direction==:right ? F.S[1] : 1.)
        end
        m.Cores[i]=U.split_left(L);m.Cores[i+1]=U.split_right(R)
        m.ort_center=direction==:right ? i+1 : i
    end
end
@testset "Componentwise gradients and complete projected sweeps" begin
    for fixture_index in (1,2,3,6)
        f=fixtures()[fixture_index];X=space(size(f.A,2))
        # Train on all supported points, with unequal, deliberately unnormalized weights.
        m=randomized!(build(f));a=amplitudes(m,X);T=X[:,findall(!iszero,a)]
        p=collect(1.:size(T,2));p .*= 1.7/sum(p)
        ref=deepcopy(m);actual=deepcopy(m);alpha=1e-4
        for direction in (:right,:left)
            reference_sweep!(ref,T,p,alpha,direction)
            step! = direction==:right ? U.sweep_right_nondeg! : U.sweep_left_nondeg!
            step!(actual,T,p,U.TrainParams(alpha,1))
            @test isapprox(amplitudes(actual,X),amplitudes(ref,X);atol=2e-8,rtol=2e-7)
            assert_canonical(actual)
        end
        @test U.Loss(actual,T,p)<U.Loss(m,T,p)
        scaled=deepcopy(m)
        for B in values(scaled.Cores[scaled.ort_center].Blocks);B .*= 3.7;end
        @test isapprox(U.Loss(scaled,T,p),U.Loss(m,T,p);atol=2e-11)
        # Dense core gradients: each physical component, both reshape conventions.
        G,qs=dense_cores(m);ids=sampleids(T)
        for j in eachindex(G),idx in CartesianIndices(G[j])
            G[j][idx]==0 && continue
            original=G[j][idx]
            G[j][idx]=original+1e-6;ap=[amplitude(G,x) for x in eachcol(X)]
            G[j][idx]=original-1e-6;am=[amplitude(G,x) for x in eachcol(X)]
            G[j][idx]=original
            saved=G[j];G[j]=zeros(size(saved));G[j][idx]=1.
            derivative=[amplitude(G,x) for x in eachcol(X)];G[j]=saved
            analytic=2sum(p)*dot(a,derivative)/sum(abs2,a)-2sum(p.*derivative[ids]./a[ids])
            fd=(dense_loss(ap,ids,p)-dense_loss(am,ids,p))/2e-6
            @test isapprox(fd,analytic;atol=3e-7,rtol=3e-6)
            # Perturb the corresponding flattened library component; compare all amplitudes.
            ql=qs[j][idx[1]];qr=qs[j+1][idx[3]];s=m.A[:,j]*(idx[2]-1)
            phys=m.Cores[j].Indices[2].InvXindex[idx[2]-1][2]
            for side in (:left,:right)
                M=side==:left ? U.collect_left(m.Cores[j]) : U.collect_right(m.Cores[j])
                q=side==:left ? qr : ql
                sector=findfirst(==(s),M.DegeneracyDict[q]);offset=M.XboundsDict[q][sector]+phys
                location=side==:left ? CartesianIndex(offset,1) : CartesianIndex(1,offset)
                M.Blocks[q][location]+=1e-6
                copy_m=deepcopy(m);copy_m.Cores[j]=side==:left ? U.split_left(M) : U.split_right(M)
                @test isapprox(amplitudes(copy_m,X),ap;atol=1e-11)
            end
        end
        # Repetition with split weights represents exactly the same target distribution.
        one=deepcopy(m);two=deepcopy(m)
        U.train_nondeg!(one,1,T,p,U.TrainParams(alpha,1))
        U.train_nondeg!(two,1,hcat(T,T),vcat(p/2,p/2),U.TrainParams(alpha,1))
        @test isapprox(amplitudes(one,X),amplitudes(two,X);atol=1e-11)
    end
end
@testset "Interpretable learning: categorical target" begin
    f=(;A=ones(Int,1,2),b=[1],T=[1 0;0 1]);m=build(f)
    U.orthogonalize!(m);U.normalize!(m);p=[.8,.2]
    U.train_nondeg!(m,60,f.T,p,U.TrainParams(.02,1));U.normalize!(m)
    @test isapprox([m[x]^2 for x in eachcol(f.T)],p;atol=1e-3)
end
