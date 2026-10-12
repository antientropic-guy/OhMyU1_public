# Check every InvXindex consumer against direct block lookup. Distinct entries
# reveal mistaken physical positions that all-one indicator blocks would hide.
function audit_core_lookup(m)
    for (site_number,c) in enumerate(m.Cores)
        site=c.Indices[2]
        for x in site.FlatDomain
            q=site.A_vec*x
            same_charge=[v for v in site.FlatDomain if site.A_vec*v==q]
            @test site.InvXindex[x]==(q,findfirst(==(x),same_charge))
        end
        for (number,B) in enumerate(values(c.Blocks)),idx in CartesianIndices(B)
            B[idx]=10000site_number+100number+10idx[2]+idx[1]+idx[3]/10
        end
        scratch=zeros(Int,length(site.A_vec))
        for left in c.Indices[1].Charges, x in site.FlatDomain
            physical=site.A_vec*x;right=left-physical
            B=get(c.Blocks,(left,physical,right),nothing)
            same_charge=[v for v in site.FlatDomain if site.A_vec*v==physical]
            position=findfirst(==(x),same_charge)
            expected=isnothing(B) ? zeros(1,1) : B[:,position,:]
            @test c[left,x]==expected
            @test c[x,right]==expected
            @test U.getval(c,left,x)==expected[1,1]
            @test U.getval(c,x,right)==expected[1,1]
            @test U.getval(scratch,c,left,x)==expected[1,1]
            @test U.getval(scratch,c,x,right)==expected[1,1]
        end
        # Matrix-valued accessors must preserve nontrivial bond dimensions too.
        generic=deepcopy(c)
        for (key,B) in generic.Blocks
            generic.Blocks[key]=reshape(collect(1.:6size(B,2)),2,size(B,2),3)
        end
        for ((left,physical,right),B) in generic.Blocks
            same_charge=[v for v in site.FlatDomain if site.A_vec*v==physical]
            for (position,x) in enumerate(same_charge)
                @test generic[left,x]==B[:,position,:]
                @test generic[x,right]==B[:,position,:]
            end
        end
    end
end
@testset "Physical-sector positions in all core accessors" begin
    for f in fixtures();audit_core_lookup(build(f));end
    # A zero column with three physical states, plus two nonzero columns.
    domain=[-1,0,2];A=[1 0 -1];b=[0]
    T=reduce(hcat,([x,y,x] for x in domain for y in domain))
    opt=U.OptimizationProblem(A=A,b=b,cost_function=x->0.,
        x_domains=[copy(domain) for _ in 1:3],site_card=fill(3,3))
    m=U.build_mps_from_feasible_samples(opt,T,1,false)
    audit_core_lookup(m)
end

@testset "Cached flattened physical positions" begin
    for f in fixtures()
        m=build(f)
        for c in m.Cores, collect_core in (U.collect_left,U.collect_right)
            leftward=collect_core===U.collect_left
            # Exercise different dictionary insertion orders explicitly.
            for order in (collect(keys(c.Blocks)),reverse(collect(keys(c.Blocks))))
                copy_c=deepcopy(c);empty!(copy_c.Blocks)
                for key in order;copy_c.Blocks[key]=copy(c.Blocks[key]);end
                matrix=collect_core(copy_c)
                for (q,charges) in matrix.DegeneracyDict
                    for x in c.Indices[2].FlatDomain
                        physical=c.Indices[2].A_vec*x
                        sector=findfirst(==(physical),charges)
                        if isnothing(sector)
                            @test !haskey(matrix.Xpositions,(q,x))
                            @test_throws ErrorException U.nondeg_merged_index(matrix,q,c.Indices[2],x)
                        else
                            domain=[v for v in c.Indices[2].FlatDomain if c.Indices[2].A_vec*v==physical]
                            within=findfirst(==(x),domain)
                            expected=matrix.XboundsDict[q][sector]+within
                            @test U.nondeg_merged_index(matrix,q,c.Indices[2],x)==expected
                        end
                    end
                end
            end
        end
    end
end
