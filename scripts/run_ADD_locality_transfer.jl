#!/usr/bin/env julia
# Reuse the frozen locality grid and paired ADD implementation; never regenerate data.
include("run_PartitionLocality.jl")

function transfer_tasks()
    # Interleave locality levels so each worker receives a range of ranks.
    [(w=w,m=m,n=n,i=i) for i in 1:15 for m in 18:20 for n in 13:15 for w in LOCALITY_SPANS]
end
function transfer_main(args=ARGS)
    options=Dict("job"=>"0","worker"=>"1","workers"=>"32")
    smoke=false;check=false
    for a in args
        if a=="--smoke";smoke=true
        elseif a=="--check";check=true
        else
            startswith(a,"--") && occursin('=',a) || error("Use --job=0 --worker=1 --workers=32, --check or --smoke")
            k,v=split(a[3:end],'=';limit=2);haskey(options,k) || error("Unknown option: $k");options[k]=v
        end
    end
    job=parse(Int,options["job"]);worker=parse(Int,options["worker"]);workers=parse(Int,options["workers"])
    @assert job in 0:3 && 1<=worker<=workers<=32
    @assert PARTITION_ADD_STRATEGY=="best nonzero mixed" && PARTITION_ADD_WEIGHT==0.4
    tasks=transfer_tasks();@assert length(tasks)==945 && allunique(tasks)
    partitions=[tasks[(j*workers+k):(4workers):end] for j in 0:3 for k in 1:workers]
    @assert length(vcat(partitions...))==945 && Set(vcat(partitions...))==Set(tasks)
    rows=load(joinpath(LOCALITY_GRID_ROOT,"manifest.jld2"),"rows")
    @assert length(rows)==945
    expected=Dict((r.w,r.m,r.n,r.i)=>r.sha256 for r in rows)
    assigned=check ? tasks : smoke ? [first(tasks)] : tasks[(job*workers+worker):(4workers):end]
    ownhash=Dict("scripts/run_ADD_locality_transfer.jl"=>bytes2hex(sha256(read(@__FILE__))))
    hashes=merge(locality_code_hashes(),ownhash)
    signature=bytes2hex(sha256(join(k*":"*hashes[k] for k in sort(collect(keys(hashes))))))[1:16]
    output=joinpath(LOCALITY_GRID_ROOT,smoke ? "smoke_add_transfer_v3" : "results_add_transfer_v3",signature)
    for t in assigned
        path=grid_path(t.w,t.m,t.n,t.i)
        @assert bytes2hex(sha256(read(path)))==expected[(t.w,t.m,t.n,t.i)]
        if check
            d=load(path);p=d["instance"];X=Int.(d["initial"])
            @assert size(p.A)==(t.m,t.n^2) && size(X)==(t.n^2,500)
            @assert all(x->x in (0,1),X) && all(p.A*X .== p.b)
            @assert (d["specification"].span,d["specification"].elements,d["specification"].n,d["specification"].i)==(t.w,t.m,t.n,t.i)
        else
            run_locality(t.w;objective="entropic",smoke,grid=true,elements=t.m,n=t.n,instances=t.i:t.i,
                modes=("ADD",),output_root=output,shard_suffix="_i$(t.i)",extra_hashes=ownhash)
        end
    end
    println(check ? "Validated 945 frozen inputs and disjoint four-job dispatch. No optimization performed." : "Completed $(length(assigned)) pairs. Output: $output")
end
abspath(PROGRAM_FILE)==(@__FILE__) && transfer_main()
