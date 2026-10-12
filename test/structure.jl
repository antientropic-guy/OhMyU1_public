@testset "Structural encoding: exhaustive components" begin
    for f in fixtures()
        @testset "$(f.name)" begin
            m=build(f);X=space(size(f.A,2));expected=closure(f)
            @test all(f.A*f.T .== f.b)
            @test [m[x] for x in eachcol(X)] == Float64.(expected)
            @test amplitudes(m,X)==Float64.(expected)
            Q=observed_charges(f.A,f.b,f.T)
            @test [Set(Tuple.(q.Charges)) for q in m.LinkIndices]==Q
            for (j,c) in enumerate(m.Cores)
                keys_ref=Set((l,Tuple(f.A[:,j]*x),r) for l in Q[j] for r in Q[j+1] for x in 0:1 if collect(l)-f.A[:,j]*x==collect(r))
                @test Set((Tuple(l),Tuple(s),Tuple(r)) for (l,s,r) in keys(c.Blocks))==keys_ref
                @test all(all(==(1),B) for B in values(c.Blocks))
                @test c.Indices[1].Card==length(c.Indices[1].Charges)
                @test c.Indices[3].Card==length(c.Indices[3].Charges)
                @test U.split_left(U.collect_left(c)).Blocks==c.Blocks
                @test U.split_right(U.collect_right(c)).Blocks==c.Blocks
            end
            # Dataset order, duplicates and column permutations must not change semantics.
            T=hcat(f.T[:,end:-1:1],f.T[:,1:1])
            duplicate_m=build(f;T)
            @test [duplicate_m[x] for x in eachcol(X)]==Float64.(expected)
            perm=reverse(1:size(f.A,2));g=(;A=f.A[:,perm],b=f.b,T=f.T[perm,:])
            permuted_m=build(g)
            @test all(permuted_m[x[perm]]==m[x] for x in eachcol(X))
        end
    end
end
@testset "Orthogonalization and normalization" begin
    for f in fixtures()
        m=build(f);X=space(size(f.A,2));a=amplitudes(m,X)
        U.orthogonalize!(m);assert_canonical(m)
        @test isapprox(amplitudes(m,X),a;atol=1e-11)
        @test isapprox(U.u1_norm(m)^2,sum(abs2,a);atol=1e-10)
        U.normalize!(m);assert_canonical(m)
        @test isapprox(sum(abs2,amplitudes(m,X)),1.;atol=1e-11)
        a=amplitudes(m,X);U.orthogonalize!(m);U.normalize!(m)
        @test isapprox(amplitudes(m,X),a;atol=1e-11)
    end
end
@testset "EELS fixed hidden-charge fixture" begin
    f=first(fixtures());X=space(4)
    # sum(x)=2; training states 1100 and 0011 omit residual charge 1 at cut 2.
    for T in (f.T,f.T[:,end:-1:1],hcat(f.T,f.T))
        m=build(f;T);before=[m[x] for x in eachcol(X)]
        @test before==[k in (3,12) ? 1. : 0. for k in 0:15]
        @test !([1] in m.LinkIndices[3].Charges)
        U.diversify!(m)
        @test [1] in m.LinkIndices[3].Charges
        after=[m[x] for x in eachcol(X)]
        @test count(!iszero,after)>2
        @test all(after.>=before)
        @test all(iszero(after[j]) || f.A*X[:,j]==f.b for j in axes(X,2))
        @test all(q.Card==length(q.Charges) for q in m.LinkIndices)
        U.orthogonalize!(m)
        @test isapprox(U.u1_norm(m)^2,count(!iszero,after);atol=1e-10)
    end
end
