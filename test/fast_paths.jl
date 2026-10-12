@testset "Dominant singular triplet versus LAPACK" begin
    rng = MersenneTwister(194)
    for T in (Float32,Float64), dims in ((1,1),(1,7),(7,1),(2,2),(3,4))
        matrices = [zeros(T,dims...),randn(rng,T,dims...),ones(T,dims...)]
        dims == (2,2) && append!(matrices,[T[1 0;0 1],T[0 -2;2 0],T[1 1;1 1+eps(T)]])
        for B in matrices, scale in (T(1),floatmax(T)/100,floatmin(T)*100)
            A = B .* scale
            u,s,v = U.dominant_triplet(A)
            F = svd(A)
            @test isapprox(s,F.S[1];rtol=30eps(T),atol=floatmin(T))
            @test isapprox(norm(u),1;rtol=30eps(T))
            @test isapprox(norm(v),1;rtol=30eps(T))
            # Singular vectors are not unique at repeated singular values.
            @test isapprox(norm(A/scale-u*(s/scale)*v'),norm(B-F.U[:,1]*(F.S[1]/scale)*F.Vt[1,:]');atol=100eps(T),rtol=100eps(T))
        end
    end
    @test_throws ArgumentError U.TrainParams(.1,1;backend=:unknown)
end

@testset "Direct nondegenerate canonicalization and training" begin
    for f in fixtures()
        m = build(f); rng = MersenneTwister(72)
        for c in m.Cores,B in values(c.Blocks); B .= randn(rng,size(B)...); end
        X = space(size(f.A,2)); before = amplitudes(m,X)
        ref = deepcopy(m)
        U.orthogonalize!(m); U.orthogonalize!(ref;backend=:lapack)
        @test isapprox(amplitudes(m,X),before;atol=1e-11,rtol=1e-11)
        @test isapprox(amplitudes(m,X),amplitudes(ref,X);atol=1e-11,rtol=1e-11)
        assert_canonical(m)
        U.normalize!(m); U.normalize!(ref)
        p = fill(1/size(f.T,2),size(f.T,2))
        U.train_nondeg!(m,2,f.T,p,U.TrainParams(.0001,1))
        U.train_nondeg!(ref,2,f.T,p,U.TrainParams(.0001,1;backend=:lapack))
        @test isapprox(abs2.(amplitudes(m,X)),abs2.(amplitudes(ref,X));atol=2e-10,rtol=2e-9)
        repeated = deepcopy(m); unsplit = deepcopy(m)
        U.train_nondeg!(repeated,1,hcat(f.T,f.T),vcat(p/2,p/2),U.TrainParams(.0001,1))
        U.train_nondeg!(unsplit,1,f.T,p,U.TrainParams(.0001,1))
        @test isapprox(abs2.(amplitudes(repeated,X)),abs2.(amplitudes(unsplit,X));atol=2e-10,rtol=2e-9)
        # A completely zero tensor still has an isometric right environment.
        z = build(f)
        for B in values(z.Cores[end].Blocks); fill!(B,0); end
        U.orthogonalize!(z)
        @test all(iszero,amplitudes(z,X))
        assert_canonical(z)
    end
    # Unequal domain sizes and physical degeneracy at a zero column.
    A = zeros(Int,1,2); b = [0]; domains = [[-1,0,2],[0,1]]
    X = reduce(hcat,([x,y] for x in domains[1] for y in domains[2]))
    opt = U.OptimizationProblem(A=A,b=b,cost_function=x->sum(x),x_domains=domains,site_card=[3,2])
    model = U.build_mps_from_feasible_samples(opt,X,1)
    U.orthogonalize!(model); U.normalize!(model)
    samples = zeros(Int,2,12000)
    Random.seed!(311)
    U.sample_nondeg_parallel!(model,samples,size(samples,2))
    @test columns(samples) == columns(X)
    @test all(abs(count(==(x),eachcol(samples))/size(samples,2)-1/6)<.02 for x in eachcol(X))
end

@testset "Integer sampling transitions reproduce dense Born probabilities" begin
    for f in fixtures()
        m = randomized!(build(f)); X = space(size(f.A,2))
        table = U.nondeg_sampling_table(m)
        expected = abs2.(amplitudes(m,X)); expected ./= sum(expected)
        function table_probability(x)
            state = table.initial_state; p = 1.
            for j in eachindex(x)
                k = findfirst(==(x[j]),table.domains[j])
                cdf = table.cumulative[j]
                p *= cdf[k,state] - (k==1 ? 0 : cdf[k-1,state])
                iszero(p) && return 0.
                state = table.next_state[j][k,state]
            end
            p
        end
        @test isapprox(map(table_probability,eachcol(X)),expected;atol=1e-12)
        for backend in (:table,:legacy)
            samples = zeros(Int,size(X,1),128)
            U.sample_nondeg!(m,samples,128;backend)
            @test all(f.A*x==f.b for x in eachcol(samples))
        end
        @test_throws ArgumentError U.sample_nondeg!(m,similar(f.T),-1)
        @test_throws DimensionMismatch U.sample_nondeg!(m,zeros(Int,1,1),2)
    end
end

@testset "Chunk-owned graph-distance buffers" begin
    for fixture in fixtures()
        model = build(fixture); candidates = space(size(fixture.A,2))
        history = [copy(q.Charges) for q in model.LinkIndices[2:end]]
        @test U.graph_distances(candidates,model,history) == [U.graph_dist_global(x,model,history) for x in eachcol(candidates)]
    end
    # Exercise signed intermediate charges and large constraint counts.
    for rows in (64,65)
        A = ones(Int,rows,2); b = ones(Int,rows); T = [1 0;0 1]
        model = U.build_mps_from_feasible_samples(problem(A,b),T,1)
        history = [copy(q.Charges) for q in model.LinkIndices[2:end]]
        candidates = [0 1 1 2 -1;0 1 -1 -1 2]
        @test U.graph_distances(candidates,model,history) == [U.graph_dist_global(x,model,history) for x in eachcol(candidates)]
    end
    f = fixtures()[1]; m = build(f); X = space(size(f.A,2))
    memory = [copy(q.Charges) for q in m.LinkIndices[2:end]]
    expected = [U.graph_dist_global(x,m,memory) for x in eachcol(X)]
    @test U.graph_distances(X,m,memory) == expected
    @test isempty(U.graph_distances(X[:,1:0],m,memory))
    calls = Threads.Atomic{Int}(0)
    callback = (buf,x,m,mem)->begin
        Threads.atomic_add!(calls,1)
        U.graph_dist_global!(buf,x,m,mem)
    end
    @test U.graph_distances(X,m,memory,callback) == expected
    @test calls[] == size(X,2)
    results = Vector{Vector{Int}}(undef,4)
    @sync for k in 1:4
        Threads.@spawn results[k] = U.graph_distances(X,m,memory)
    end
    @test all(==(expected),results)
end
