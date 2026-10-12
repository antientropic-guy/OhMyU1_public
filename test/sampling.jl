@testset "Sampling after training" begin
    for f in fixtures()
        m=build(f);U.orthogonalize!(m);U.normalize!(m)
        p=collect(1.:size(f.T,2));p./=sum(p)
        U.train_nondeg!(m,2,f.T,p,U.TrainParams(.01,1));U.normalize!(m)
        X=space(size(f.A,2));G,_=dense_cores(m);amps=[amplitude(G,x) for x in eachcol(X)]
        # Verify exact conditional probabilities independently of empirical draws.
        for (h,x) in enumerate(eachcol(X))
            q=copy(f.b);prob=1.
            for j in eachindex(m.Cores)
                c=m.Cores[j];vals=[U.getval(c,q,v)^2 for v in 0:1]
                if sum(vals)==0;prob=0.;break;end
                prob*=vals[x[j]+1]/sum(vals);q-=f.A[:,j]*x[j]
            end
            @test isapprox(prob,amps[h]^2;atol=2e-11)
        end
        for parallel in (false,true)
            sample! = parallel ? U.sample_nondeg_parallel! : U.sample_nondeg!
            Y=zeros(Int,size(f.A,2),512);Random.seed!(94);sample!(m,Y,512)
            @test all(f.A*Y .== f.b)
            @test all(x->amps[only(sampleids(reshape(x,:,1)))]!=0,eachcol(Y))
            @test isapprox(amplitudes(m,X),amps;atol=1e-12)
            if !parallel
                Z=similar(Y);Random.seed!(94);sample!(m,Z,512);@test Y==Z
            end
        end
    end
end
@testset "Actual draws follow Born probabilities" begin
    f=(;A=ones(Int,1,2),b=[1],T=[1 0;0 1]);m=build(f)
    U.orthogonalize!(m);U.normalize!(m)
    U.train_nondeg!(m,40,f.T,[.8,.2],U.TrainParams(.02,1));U.normalize!(m)
    target=m[[1,0]]^2;N=16384
    for sampler! in (U.sample_nondeg!,U.sample_nondeg_parallel!)
        X=zeros(Int,2,N);Random.seed!(1559);sampler!(m,X,N)
        @test all(sum(X;dims=1).==1)
        # Very conservative binomial envelope; exact conditionals are tested
        # separately, so correctness does not rest on a statistical test alone.
        @test abs(sum(X[1,:])/N-target)<=8sqrt(target*(1-target)/N)+2/N
    end
end
