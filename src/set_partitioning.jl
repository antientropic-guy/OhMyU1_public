"""
    strip_partition_instance(rng, n; width=3, variables=n^2, density=:balanced)

Random exact-cover problem on a width × n strip. Columns are distinct connected
tiles of 1–4 cells, spanning at most three strip columns. Singletons and a
domino spanning tree are mandatory. Remaining tiles are sampled without
replacement, with size weights exp(beta*(size-2)), beta=-2,0,2 for
`:light`, `:balanced`, `:heavy`. These names describe a size bias, not a
prescribed density. Both row and variable order are randomly permuted.
Returns plain arrays suitable for JLD2 storage. Cell IDs in `tiles` retain
geometric order; `row_order` maps the rows of A back to these cell IDs.
"""
function strip_partition_instance(rng::AbstractRNG, n::Int;
        width::Int=3, variables::Int=n^2, density::Symbol=:balanced)
    width > 0 && n > 0 && width*n <= 128 || throw(ArgumentError("Require positive dimensions and width*n <= 128"))
    beta = density == :light ? -2.0 : density == :balanced ? 0.0 :
        density == :heavy ? 2.0 : throw(ArgumentError("Unknown density"))
    m = width*n
    bit(u) = UInt128(1) << (u-1)
    neighbours = [Int[] for _ in 1:m]
    for u in 1:m
        r, c = mod1(u,width), cld(u,width)
        r > 1 && push!(neighbours[u],u-1)
        r < width && push!(neighbours[u],u+1)
        c > 1 && push!(neighbours[u],u-width)
        c < n && push!(neighbours[u],u+width)
    end
    level = Set(bit(u) for u in 1:m)
    catalogue = copy(level)
    for _ in 2:4
        next = Set{UInt128}()
        for mask in level, u in 1:m
            mask & bit(u) == 0 && continue
            for v in neighbours[u]
                mask & bit(v) != 0 && continue
                new = mask | bit(v)
                lo = cld(trailing_zeros(new)+1,width)
                hi = cld(128-leading_zeros(new),width)
                hi-lo <= 2 && push!(next,new)
            end
        end
        union!(catalogue,next)
        level = next
    end
    mandatory = Set(bit(u) for u in 1:m)
    # A connected backbone: vertical edges, and horizontal edges on the top row.
    for c in 1:n, r in 1:width-1
        u = (c-1)*width+r
        push!(mandatory,bit(u)|bit(u+1))
    end
    for c in 1:n-1
        u = (c-1)*width+1
        push!(mandatory,bit(u)|bit(u+width))
    end
    length(mandatory) <= variables <= length(catalogue) ||
        throw(ArgumentError("variables must be in $(length(mandatory)):$(length(catalogue))"))
    candidates = sort!(collect(setdiff(catalogue,mandatory)))
    # Exponential races implement weighted sampling without replacement.
    keys = [-log(rand(rng))/exp(beta*(count_ones(t)-2)) for t in candidates]
    chosen = vcat(sort!(collect(mandatory)), candidates[sortperm(keys)[1:variables-length(mandatory)]])
    shuffle!(rng,chosen)
    tiles = [[u for u in 1:m if t & bit(u) != 0] for t in chosen]
    row_order = randperm(rng,m)
    A = [Int(u in tile) for u in row_order, tile in tiles]
    (; A, b=ones(Int,m), tiles, row_order, width, n, density=String(density),
       actual_density=sum(A)/length(A), column_sizes=vec(sum(A;dims=1)))
end

"""
    ExactPartitionSampler(tiles, number_of_cells; max_states=2_000_000)

Count ALL exact covers using BigInt dynamic programming. At each state choose
the first uncovered cell and branch over disjoint tiles containing it.
Every exact cover has exactly one such path. The memo contains completion
counts, not a list of feasible points. It is read-only after construction and
can be shared between tasks; each task must supply its own RNG/output buffer.
No model, floating-point branch probabilities, rejection of covers, or MCMC.
"""
struct ExactPartitionSampler
    masks::Vector{UInt128}
    by_cell::Vector{Vector{Int}}
    full::UInt128
    counts::Dict{UInt128,BigInt}
end

function ExactPartitionSampler(tiles::AbstractVector, m::Int; max_states::Int=2_000_000)
    1 <= m <= 128 || throw(ArgumentError("Require 1 <= cells <= 128"))
    masks = UInt128[]
    by_cell = [Int[] for _ in 1:m]
    for (j,tile) in enumerate(tiles)
        !isempty(tile) && all(u -> 1 <= u <= m,tile) && allunique(tile) ||
            throw(ArgumentError("Invalid tile"))
        push!(masks,foldl(|,(UInt128(1) << (u-1) for u in tile);init=UInt128(0)))
        foreach(u -> push!(by_cell[u],j),tile)
    end
    full = typemax(UInt128) >> (128-m)
    counts = Dict{UInt128,BigInt}(full=>big(1))
    function completions(mask)
        haskey(counts,mask) && return counts[mask]
        length(counts) < max_states || error("Exact-cover DP exceeded max_states=$max_states")
        u = trailing_zeros(full & ~mask)+1
        total = big(0)
        for j in by_cell[u]
            t = masks[j]
            mask & t == 0 && (total += completions(mask|t))
        end
        counts[mask] = total
        total
    end
    completions(UInt128(0)) > 0 || throw(ArgumentError("No exact cover exists"))
    ExactPartitionSampler(masks,by_cell,full,counts)
end

# Exact uniform integer in [0,limit); explicit bit rejection avoids Float64
# rounding even when the number of covers is greater than 2^53.
function partition_randbelow(rng::AbstractRNG, limit::BigInt)
    limit > 0 || throw(ArgumentError("Positive limit required"))
    bits = ndigits(limit-1;base=2)
    words = cld(bits,64)
    while true
        value = big(0)
        for _ in 1:words
            value = (value << 64) + rand(rng,UInt64)
        end
        value >>= 64words-bits
        value < limit && return value
    end
end

"""Fill columns of X with independent uniform exact covers, with replacement.
Works on views; clears previous values. The column order is `tiles` order.
"""
function fill_partition_vectors!(rng::AbstractRNG, X::AbstractMatrix{<:Integer},
        sampler::ExactPartitionSampler)
    size(X,1) == length(sampler.masks) || throw(DimensionMismatch("Wrong variable count"))
    fill!(X,0)
    for column in axes(X,2)
        mask = UInt128(0)
        while mask != sampler.full
            u = trailing_zeros(sampler.full & ~mask)+1
            draw = partition_randbelow(rng,sampler.counts[mask])
            selected = false
            for j in sampler.by_cell[u]
                tile = sampler.masks[j]
                mask & tile != 0 && continue
                weight = sampler.counts[mask|tile]
                if draw < weight
                    X[j,column] = 1
                    mask |= tile
                    selected = true
                    break
                end
                draw -= weight
            end
            selected || error("Corrupt exact-cover counts")
        end
    end
    X
end
