import Base: getindex
# Index types;
abstract type IndexType end

mutable struct ChargeIndex{S<:Integer} <: IndexType
    Charges::Set{Vector{S}}
    Card::Int

    ChargeIndex(charges::Set{Vector{S}}) where {S<:Integer} = begin
        card = length(charges)
        new{S}(charges, card)
    end
end

mutable struct SiteIndex{S<:Integer} <: IndexType
    Charges::Vector{Vector{S}}
    Domain::Vector{Vector{S}}
    FlatDomain::Vector{S}
    A_vec::Vector{S}
    InvXindex::Dict{S, Tuple{Vector{S}, S}}
end


"""
    compute_link_charges(A::Matrix{S}, b::Vector{S}, T::Matrix{S}) where {S<:Integer}

Compute U1-symmetric tensor train link charges from the dataset and constraints.

# Arguments
- `A::Matrix{S}`: Constraint matrix of size `(num_constraints, d)`.
- `b::Vector{S}`: Right-hand side vector.
- `T::Matrix{S}`: Feasible samples (by columns).

# Returns
- `Array{S,3}`: 3D array of link charges with shape `(d+1, num_constraints, num_samples)`.

# Notes
Initializes edge charges `(b, 0)` and computes cumulative deltas using Einstein summation.
"""
function compute_link_charges(A::Matrix{S}, b::Vector{S}, T::Matrix{S}) where {S<:Integer}
    num_constraints, d = size(A)                     
    num_samples = size(T, 2)

    link_charges = Array{S}(undef, d+1, num_constraints, num_samples)

    for j in 1:num_samples
        link_charges[1, :, j] .= b
    end

    for j in 1:num_samples
        for i in 1:d
            for k in 1:num_constraints
                link_charges[i+1, k, j] = link_charges[i, k, j] - A[k, i] * T[i, j]
            end
        end
    end
    return link_charges
end

"""
    compute_site_charges(A::Matrix{T}, x_domains::Vector{Vector{T}}) where {T<:Integer}

Compute site charges for each domain of variables.

# Arguments
- `A::Matrix{T}`: Constraint matrix.
- `x_domains::Vector{Vector{T}}`: Domains for each variable.

# Returns
- `Vector{Array{T}}`: Vector of arrays, each containing site charges for one domain.

# Notes
Each domain is sorted before computation; charges are calculated as `domain[i] * A[:, j]`.
"""
function compute_site_charges(A::Matrix{T}, x_domains::Vector{Vector{T}}) where {T<:Integer}
    num_constraints, d = size(A)
    site_charges = Vector{Array{T}}()
    for j in eachindex(x_domains)
        sort!(x_domains[j])
        domain = x_domains[j]
        domain_charges = Array{T}(undef, num_constraints, length(domain))
        for i in eachindex(domain)
            domain_charges[:, i] .= domain[i] .* A[:, j]
        end
        push!(site_charges, domain_charges)
    end
    return site_charges
end

"""
    compute_indices(link_charges::Array{T,3}, site_charges::Vector{Array{T}}, flux::Vector{T}, x_domains::Vector{Vector{T}}, A::Matrix{T}) where {T<:Integer}

Construct index structures for link and site charges.

# Arguments
- `link_charges::Array{T,3}`: 3D array of link charges.
- `site_charges::Vector{Array{T}}`: Vector of site charge arrays.
- `flux::Vector{T}`: Initial flux vector.
- `x_domains::Vector{Vector{T}}`: Domains of variables.
- `A::Matrix{T}`: Constraint matrix.

# Returns
- `(Vector{ChargeIndex}, Vector{SiteIndex})`: Tuple of link indices and site indices.

# Notes
Handles special case when a column of `A` is zero.
"""
function compute_indices(link_charges::Array{T, 3}, 
        site_charges::Vector{Array{T}}, 
        flux::Vector{T}, 
        x_domains::Vector{Vector{T}}, 
        A::Matrix{T}) where {T<:Integer}
    
    d = length(site_charges)
    N_constr = length(flux)
    zero_charge = zeros(T, N_constr)
    
    first_index = ChargeIndex(Set([flux]))
    link_indices = [first_index]
    site_indices = Vector{SiteIndex{T}}()
    
    for i in 1:d-1
        link_i_index_charges = Set(Vector(c) for c in eachcol(link_charges[i+1, :, :]))
        link_i = ChargeIndex(link_i_index_charges)
        push!(link_indices, link_i)
    end
    
    for i in 1:d
        unique_site_charges_i = unique(eachcol(site_charges[i]))
        site_i_index_charges = [Vector(v) for v in unique_site_charges_i]
        A_vec = A[:, i]

        inv_x_dict = Dict{T, Tuple{Vector{T}, T}}()
        
        if length(unique_site_charges_i) == size(site_charges[i], 2)
            for x in x_domains[i]
                inv_x_dict[x] = (A_vec * x, 1)
            end
            site_i = SiteIndex{T}(site_i_index_charges, [[x_domains[i][j]] for j in 1:length(x_domains[i])], x_domains[i], A_vec, inv_x_dict)
        else
            # Case with 0 column of the matrix A:
            println("Zero column")
            for (j, x) in enumerate(x_domains[i])
                inv_x_dict[x] = (zero_charge, j)
            end
            site_i = SiteIndex{T}(site_i_index_charges, [x_domains[i]], x_domains[i], A_vec, inv_x_dict)  
        end
        push!(site_indices, site_i)
            
    end
    last_index = ChargeIndex(Set([zero_charge]))
    push!(link_indices, last_index)
    return link_indices, site_indices
end

# U1-symmetric tensors types:
abstract type AbstractU1Tensor end

mutable struct U1Core{S<:Integer, N<:AbstractFloat} <: AbstractU1Tensor
    Blocks::Dict{Tuple{Vector{S}, Vector{S}, Vector{S}}, Array{N}}
    Indices::Tuple{ChargeIndex{S}, SiteIndex{S}, ChargeIndex{S}}
    Flux::Vector{S}
    mem_buf::Vector{S}
    return_const::Matrix{S}
end

function U1Core(blocks::Dict{Tuple{Vector{S}, Vector{S}, Vector{S}}, Array{N}}, indices::Tuple{ChargeIndex{S}, SiteIndex{S}, ChargeIndex{S}}, flux::Vector{S}) where {S<:Integer, N<:AbstractFloat}
    mem_buf = Vector{S}(undef, length(flux))
    return_const = [N(0.0);;]
    U1Core{S,N}(blocks, indices, flux, mem_buf, return_const)
end

mutable struct U1MPS{S<:Integer, N<:AbstractFloat} <: AbstractU1Tensor
    Cores::Vector{U1Core{S, N}}
    SiteIndices::Vector{SiteIndex{S}}
    LinkIndices::Vector{ChargeIndex{S}}
    A::Matrix{S}
    Flux::Vector{S}
    ort_center::Int
end

function U1MPS(cores::Vector{U1Core{S, N}}, sites::Vector{SiteIndex{S}}, links::Vector{ChargeIndex{S}}, A, flux) where {S<:Integer, N<:AbstractFloat}
    U1MPS{S, N}(cores, sites, links, A, flux, 0)
end

mutable struct U1Matrix{S, Y}<:AbstractU1Tensor
    Blocks::Dict{Vector{S}, Array{Y}}
    Indices::Tuple{ChargeIndex{S}, SiteIndex{S}, ChargeIndex{S}}
    DegeneracyDict::Dict{Vector{S}, Vector{Vector{S}}}
    XboundsDict::Dict{Vector{S}, Vector{S}}
    SplitIndicesDict::Dict{Vector{S}, Vector{Tuple{S, S}}}
    Flux::Vector{S}
end

# Operations with U1-symmetric tensors:
function mul!(tensor::AbstractU1Tensor, scalar::Real)
    for k in keys(tensor.Blocks)
        tensor.Blocks[k] .*= scalar
    end
end

function mul!(scalar::Real, tensor::AbstractU1Tensor)
    for k in keys(tensor.Blocks)
        tensor.Blocks[k] .*= scalar
    end
end

function div!(tensor::AbstractU1Tensor, scalar::Real)
    for k in keys(tensor.Blocks)
        tensor.Blocks[k] ./= scalar
    end
end

"""
    init_u1_mps(::Type{Y}, link_indices::Vector{ChargeIndex{S}}, site_indices::Vector{SiteIndex{S}}, A::Matrix{S}, flux::Vector{S}, link_degeneracy, site_cards) where {S<:Integer, Y<:AbstractFloat}

Initialize a U1-symmetric Matrix Product State (MPS).

# Arguments
- `Y`: Floating type for tensor entries.
- `link_indices::Vector{ChargeIndex{S}}`: Link indices.
- `site_indices::Vector{SiteIndex{S}}`: Site indices.
- `A::Matrix{S}`: Constraint matrix.
- `flux::Vector{S}`: Flux vector.
- `link_degeneracy`: Degeneracy factor for links.
- `site_cards`: Cardinalities of site domains.

# Returns
- `U1MPS{S,Y}`: Initialized MPS object.

# Notes
Constructs cores with blocks consistent with charge conservation.
"""
function init_u1_mps(::Type{Y}, link_indices::Vector{ChargeIndex{S}}, 
        site_indices::Vector{SiteIndex{S}}, 
        A::Matrix{S}, 
        flux::Vector{S}, 
        link_degeneracy, 
        site_cards) where {S<:Integer, Y<:AbstractFloat}
    
    N_sites = length(site_indices)
    cores = Vector{U1Core{S, Y}}(undef, N_sites)
    N_constr = length(flux)
    zero_charge = zeros(S, N_constr)

    for i in 1:N_sites
        inds = (link_indices[i], site_indices[i], link_indices[i+1])
        cores[i] = U1Core(Dict{Tuple{Vector{S}, Vector{S}, Vector{S}}, Array{Y}}(), inds, zero_charge)
    end

    #TODO: different degeneracies
    link_degs = vcat([1], fill(link_degeneracy, length(link_indices) - 2), [1])
    
    for i in 1:N_sites
        left_index = link_indices[i]
        site_index = site_indices[i]
        site_deg = 1
        if site_cards[i] > length(site_index.Charges)
            site_deg = site_cards[i]
        end
            
        for left_charge in left_index.Charges
            for site_charge in site_index.Charges
                right_charge = left_charge .- site_charge
                if in(right_charge, link_indices[i+1].Charges)
                    cores[i].Blocks[(left_charge, site_charge, right_charge)] = ones(link_degs[i], site_deg, link_degs[i+1])
                end 
            end
        end
    end
    return U1MPS(cores, site_indices, link_indices, A, zero_charge)
end

"""
    collect_left(core::U1Core{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Reshape a U1-symmetric core by collecting the site index to the left.

# Arguments
- `core::U1Core{S,Y}`: U1 core tensor.

# Returns
- `U1Matrix{S,Y}`: Matrix representation with reshaped blocks.

# Notes
Builds dictionaries for degeneracy, bounds, and split indices to restore shape later.
"""
function collect_left(core::U1Core{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    flux = core.Flux
    indices = core.Indices
    N_constr = length(core.Flux)
    dumb_dict = Dict{Vector{S}, Array{Y}}()
    charges_degeneracy_dict = Dict{Vector{S}, Vector{Vector{S}}}()
    # Dicts needed for restoring shape:
    x_bounds_dict = Dict{Vector{S}, Vector{S}}()
    split_indices_dict = Dict{Vector{S}, Vector{Tuple{S, S}}}()

    # Dictionary for reshaping:
    for charges_tuple in keys(core.Blocks)
        left_link_charge, site_charge, charge = charges_tuple
        list_ref = get(charges_degeneracy_dict, charge, nothing)
        if list_ref === nothing
            charges_degeneracy_dict[charge] = [site_charge]
            x_bounds_dict[charge] = [0]
            split_indices_dict[charge] = Vector{Tuple{S, S}}()
        else
            push!(list_ref, site_charge)
        end
    end

    # Define blocks:
    for charge in keys(charges_degeneracy_dict)
        all_sites = charges_degeneracy_dict[charge]
        blocks = Vector{Matrix{Y}}(undef, length(all_sites))
        x_bounds_dict_charge = x_bounds_dict[charge]
        split_indices_dict_charge = split_indices_dict[charge]
        for (i, site_charge) in enumerate(all_sites)
            block = core.Blocks[(charge + site_charge, site_charge, charge)]
            s = size(block)
            reshaped_block = reshape(block, s[1]*s[2], s[3])
            blocks[i] = reshaped_block
            last_el = x_bounds_dict_charge[end]
            push!(x_bounds_dict_charge, last_el + s[1]*s[2])
            push!(split_indices_dict_charge, (s[1], s[2]))
        end
        dumb_dict[charge] = reduce(vcat, blocks)
    end
    return U1Matrix{S, Y}(dumb_dict, indices, charges_degeneracy_dict, x_bounds_dict, split_indices_dict, flux)
end

"""
    collect_right(core::U1Core{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Reshape a U1-symmetric core by collecting the site index to the right.

# Arguments
- `core::U1Core{S,Y}`: U1 core tensor.

# Returns
- `U1Matrix{S,Y}`: Matrix representation with reshaped blocks.
"""
function collect_right(core::U1Core{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    flux = core.Flux
    indices = core.Indices
    N_constr = length(core.Flux)
    dumb_dict = Dict{Vector{S}, Array{Y}}()
    charges_degeneracy_dict = Dict{Vector{S}, Vector{Vector{S}}}()
    # Dicts needed for restoring shape:
    x_bounds_dict = Dict{Vector{S}, Vector{S}}()
    split_indices_dict = Dict{Vector{S}, Vector{Tuple{S, S}}}()
    
    # Dictionary for reshaping:
    for charges_tuple in keys(core.Blocks)
        (charge, site_charge, right_link_charge) = charges_tuple
        list_ref = get(charges_degeneracy_dict, charge, nothing)
        if list_ref === nothing
            charges_degeneracy_dict[charge] = [site_charge]
            x_bounds_dict[charge] = [0]
            split_indices_dict[charge] = Vector{Tuple{S, S}}()
        else
            push!(list_ref, site_charge)
        end
    end

    # Define blocks:
    for charge in keys(charges_degeneracy_dict)
        all_sites = charges_degeneracy_dict[charge]
        x_bounds_dict_charge = x_bounds_dict[charge]
        split_indices_dict_charge = split_indices_dict[charge]
        blocks = Vector{Matrix{Y}}(undef, length(all_sites))
        for (i, site_charge) in enumerate(all_sites)
            block = core.Blocks[(charge, site_charge, charge-site_charge)]
            s = size(block)
            reshaped_block = reshape(block, s[1], s[2] * s[3])
            blocks[i] = reshaped_block
            last_el = x_bounds_dict_charge[end]
            push!(x_bounds_dict_charge, last_el + s[2]*s[3])
            push!(split_indices_dict_charge, (s[2], s[3]))
        end
        dumb_dict[charge] = reduce(hcat, blocks)
    end
    return U1Matrix{S, Y}(dumb_dict, indices, charges_degeneracy_dict, x_bounds_dict, split_indices_dict, flux)
end

"""
    split_left(m::U1Matrix{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Inverse of `collect_left`.

# Arguments
- `m::U1Matrix{S,Y}`: U1 matrix obtained from left collection.

# Returns
- `U1Core{S,Y}`: Reconstructed core tensor.
"""
function split_left(m::U1Matrix{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    #blocks, indices, flux
    indices = m.Indices
    flux = m.Flux
    N_constr = length(flux)
    blocks = Dict{Tuple{Vector{S}, Vector{S}, Vector{S}}, Array{Y}}()

    for charge in keys(m.Blocks)
        for (i, x) in enumerate(m.DegeneracyDict[charge])
            x_splits = m.XboundsDict[charge] 
            a_x_splits = m.SplitIndicesDict[charge][i]
            (a_deg, x_deg) = (a_x_splits[1], a_x_splits[2])
            (lb, ub) = (x_splits[i]+1, x_splits[i+1])
            blocks[(charge + x, x, charge)] = copy(reshape(m.Blocks[charge][lb:ub, :], a_deg, x_deg, :))
        end
    end
    return U1Core(blocks, indices, flux)
end

"""
    split_right(m::U1Matrix{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Inverse of `collect_right`.

# Arguments
- `m::U1Matrix{S,Y}`: U1 matrix obtained from right collection.

# Returns
- `U1Core{S,Y}`: Reconstructed core tensor.
"""
function split_right(m::U1Matrix{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    #blocks, indices, flux
    indices = m.Indices
    flux = m.Flux
    N_constr = length(flux)
    blocks = Dict{Tuple{Vector{S}, Vector{S}, Vector{S}}, Array{Y}}()

    for charge in keys(m.Blocks)
        for (i, x) in enumerate(m.DegeneracyDict[charge])
            x_splits = m.XboundsDict[charge] 
            x_a_splits = m.SplitIndicesDict[charge][i]
            x_deg = x_a_splits[1]
            a_deg = x_a_splits[2]
            lb = x_splits[i]+1
            ub = x_splits[i+1]
            blocks[(charge, x, charge - x)] = copy(reshape(m.Blocks[charge][:, lb:ub], :, x_deg, a_deg))
        end
    end
    return U1Core(blocks, indices, flux)
end

mutable struct MergedCores{S, Y} <: AbstractU1Tensor
    Blocks::Dict{Vector{S}, Array{Y}} 
    left_matrix::U1Matrix{S, Y}
    right_matrix::U1Matrix{S, Y}

    function MergedCores(A::U1Core{S, Y}, B::U1Core{S, Y}) where {S<:Integer, Y<:AbstractFloat}
        left_matrix = collect_left(A)
        right_matrix = collect_right(B)
        N_constr = length(A.Flux)
        blocks = Dict{Vector{S}, Array{Y}}()

        for charge in keys(left_matrix.Blocks)
            l_block = left_matrix.Blocks[charge]
            r_block = right_matrix.Blocks[charge]
            blocks[charge] = l_block * r_block
        end
        new{S, Y}(blocks, left_matrix, right_matrix)
    end
end


"""
    u1_lq(m::U1Matrix{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Perform sparse LQ decomposition of a U1 matrix.

# Arguments
- `m::U1Matrix{S,Y}`: U1 matrix.

# Returns
- `(Dict{Vector{S},Array{Y}}, Dict{Vector{S},Array{Y}})`: Tuple of L and Q blocks.
"""
function u1_lq(m::U1Matrix{S, Y}) where {S<: Integer, Y<:AbstractFloat}

    N_constr = length(m.Flux)
    blocks_Q = Dict{Vector{S}, Array{Y}}()
    blocks_L = Dict{Vector{S}, Array{Y}}()

    for charge in keys(m.Blocks)
        block = m.Blocks[charge]
        if size(block, 1) == 1
            # For a one-dimensional left bond, LQ is just normalization of a
            # row vector: a = ||a||₂ (a / ||a||₂). Avoid a LAPACK call for
            # every tiny charge sector.
            scale = norm(block)
            if iszero(scale)
                L_c, Q_c = lq(block)
                blocks_Q[charge] = Matrix(Q_c)
                blocks_L[charge] = Matrix(L_c)
            else
                blocks_L[charge] = reshape(Y[scale], 1, 1)
                blocks_Q[charge] = block ./ scale
            end
        else
            L_c, Q_c = lq(block)
            blocks_Q[charge] = Matrix(Q_c)
            blocks_L[charge] = Matrix(L_c)
        end
    end

    return blocks_L, blocks_Q
end


function _has_nondegenerate_bonds(mps::U1MPS)
    return all(block -> size(block, 1) == 1 && size(block, 3) == 1,
               Iterators.flatten(values(core.Blocks) for core in mps.Cores))
end

"""Right-to-left orthogonalization specialized for bond degeneracy one."""
function _orthogonalize_nondeg!(mps::U1MPS{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    for n in reverse(2:length(mps.Cores))
        core = mps.Cores[n]
        squared_norms = Dict{Vector{S},Y}()
        for (charges, block) in core.Blocks
            left_charge = charges[1]
            squared_norms[left_charge] = get(squared_norms, left_charge, zero(Y)) + sum(abs2, block)
        end

        norms = Dict{Vector{S},Y}()
        for (charge, squared_norm) in squared_norms
            norm_value = sqrt(squared_norm)
            iszero(norm_value) && throw(DomainError(norm_value, "zero non-degenerate MPS sector"))
            norms[charge] = norm_value
        end

        for (charges, block) in core.Blocks
            block ./= norms[charges[1]]
        end
        for (charges, block) in mps.Cores[n - 1].Blocks
            block .*= norms[charges[3]]
        end
    end
    mps.ort_center = 1
    return nothing
end

"""
    orthogonalize!(mps::U1MPS{S,Y}) where {S<:Integer, Y<:AbstractFloat}

Orthogonalize a U1-symmetric MPS in place.

# Arguments
- `mps::U1MPS{S,Y}`: MPS object.

# Returns
- `Nothing`.

# Notes
Moves orthogonality center to site 1 using successive LQ decompositions.
"""
function orthogonalize!(mps::U1MPS{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    if mps.ort_center == 1
        return nothing
    elseif _has_nondegenerate_bonds(mps)
        return _orthogonalize_nondeg!(mps)
    else
        num_sites = length(mps.Cores)
        for n in reverse(2:num_sites)
            core = mps.Cores[n]
            reshaped_core = collect_right(core)
            blocks_L, blocks_Q = u1_lq(reshaped_core)
            reshaped_core.Blocks = blocks_Q
            mps.Cores[n] = split_right(reshaped_core)
            blocks = mps.Cores[n-1].Blocks
            for charge_tuple in keys(blocks)
                c1, x, c2 = charge_tuple
                previous_block = blocks[charge_tuple]
                
                # blocks[charge_tuple] = ein"ijk,km -> ijm"(previous_block, blocks_L[c2]) #TODO: optimize
                
                A = previous_block
                B = blocks_L[c2]
                I, J, K = size(A)
                M = size(B,2)
                
                out = Array{eltype(A)}(undef, I, J, M)
                for i in 1:I
                    for j in 1:J
                        @inbounds for m in 1:M
                            s = zero(eltype(A))
                            for k in 1:K
                                s += A[i,j,k] * B[k,m]
                            end
                            out[i,j,m] = s
                        end
                    end
                end
                blocks[charge_tuple] = out

            end
        end
    end
    mps.ort_center = 1
end


"""
    u1_norm(core::U1Core)

Compute Frobenius (L2) norm of a U1 core tensor.

# Arguments
- `core::U1Core`: U1 core tensor.

# Returns
- `Float64`: Norm value.
"""
function u1_norm(core::U1Core)
    n = 0
    for block in values(core.Blocks)
        n += sum(abs2, block)
    end
    return sqrt(n)
end


"""
    u1_norm(mps::U1MPS)

Compute Frobenius norm of a U1 MPS.

# Arguments
- `mps::U1MPS`: MPS object.

# Returns
- `Float64`: Norm value.

# Notes
Requires MPS to be in orthogonal state.
"""
function u1_norm(mps::U1MPS)
    if mps.ort_center != 0
        return u1_norm(mps.Cores[mps.ort_center])
    else
        println("MPS is not in orthogonal state")
        return NaN
    end
end

"""
    normalize!(mps::U1MPS)

Normalize a U1 MPS in place.

# Arguments
- `mps::U1MPS`: MPS object.

# Returns
- `Nothing`.

# Notes
Divides the orthogonal center core by the norm.
"""
function normalize!(mps::U1MPS)
    mps_norm = u1_norm(mps)
    div!(mps.Cores[mps.ort_center], mps_norm)
end

"""
    getindex(core::U1Core{S,Y}, left_charge::Vector{S}, i::S)

Access block data of a U1 core given left charge and site index.

# Returns
- `Array{Y}`: Slice of block array corresponding to the charge tuple.
"""
function getindex(core::U1Core{S, Y}, left_charge::Vector{S}, i::S) where {S<:Integer, Y<:AbstractFloat} 
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    core.mem_buf .= left_charge .- site_charge
    temp_tuple = (left_charge, site_charge, core.mem_buf)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data
    else
        return block_data[:, deg_ind, :]
    end
end


"""
    getindex(core::U1Core{S,Y}, i::S, right_charge::Vector{S})

Access block data of a U1 core given site index and right charge.
"""
function getindex(core::U1Core{S, Y}, i::S, right_charge::Vector{S}) where {S<:Integer, Y<:AbstractFloat}
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    core.mem_buf .= right_charge .+ site_charge
    temp_tuple = (core.mem_buf, site_charge, right_charge)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data
    else
        return block_data[:, deg_ind, :]
    end
end

"""
    getindex(mps::U1MPS{S,Y}, state_indices::AbstractVector{S})

Evaluate MPS amplitude for a given sequence of site indices.

# Returns
- `Y`: Scalar value of the MPS at the specified state.
"""
function getindex(mps::U1MPS{S, Y}, state_indices::AbstractVector{S}) where {S<:Integer, Y<:AbstractFloat}
    N = length(mps.Cores)
    core1 = mps.Cores[1]
    site_ind = core1.Indices[2]
    A_vec = site_ind.A_vec
    b_charge = first(core1.Indices[1].Charges)
    mps_value = core1[b_charge, state_indices[1]]
    left_charge = b_charge - state_indices[1] * A_vec
    for i in 2:N
        core = mps.Cores[i]
        site_ind = core.Indices[2]
        A_vec = site_ind.A_vec
        mps_value = mps_value * core[left_charge, state_indices[i]]
        left_charge .-= state_indices[i] .* A_vec
    end
    return mps_value[1, 1]
end

"""
    getval(core::U1Core{S,Y}, left_charge::Vector{S}, i::S)

Access scalar value from a non-degenerate U1 core block using left charge and site index.
"""
function getval(core::U1Core{S, Y}, left_charge::Vector{S}, i::S) where {S<:Integer, Y<:AbstractFloat} 
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    core.mem_buf .= left_charge .- site_charge
    temp_tuple = (left_charge, site_charge, core.mem_buf)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data[1, 1]
    else
        return block_data[1, deg_ind, 1]
    end
end

function getval(mem_buf::Vector{S}, core::U1Core{S, Y}, left_charge::Vector{S}, i::S) where {S<:Integer, Y<:AbstractFloat} 
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    mem_buf .= left_charge .- site_charge
    temp_tuple = (left_charge, site_charge, mem_buf)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data[1, 1]
    else
        return block_data[1, deg_ind, 1]
    end
end

function getval(mem_buf::Vector{S}, core::U1Core{S, Y}, i::S, right_charge::Vector{S}) where {S<:Integer, Y<:AbstractFloat}
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    mem_buf .= right_charge .+ site_charge
    temp_tuple = (mem_buf, site_charge, right_charge)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data[1, 1]
    else
        return block_data[1, deg_ind, 1]
    end
end


"""
    getval(core::U1Core{S,Y}, i::S, right_charge::Vector{S})

Access scalar value from a non-degenerate U1 core block using site index and right charge.
"""
function getval(core::U1Core{S, Y}, i::S, right_charge::Vector{S}) where {S<:Integer, Y<:AbstractFloat}
    site_charge, deg_ind = core.Indices[2].InvXindex[i]
    core.mem_buf .= right_charge .+ site_charge
    temp_tuple = (core.mem_buf, site_charge, right_charge)
    block_data = get(core.Blocks, temp_tuple, core.return_const)
    if block_data === core.return_const
        return block_data[1, 1]
    else
        return block_data[1, deg_ind, 1]
    end
end

function _build_nondeg_sampling_tables(mps::U1MPS{S,Y}) where {S<:Integer,Y<:AbstractFloat}
    tables = Vector{Dict{Vector{S},Vector{Y}}}(undef, length(mps.Cores))
    fallbacks = Vector{Vector{Y}}(undef, length(mps.Cores))
    lookup_buffer = Vector{S}(undef, length(mps.Flux))
    for (core_index, core) in enumerate(mps.Cores)
        domain = core.Indices[2].FlatDomain
        table = Dict{Vector{S},Vector{Y}}()
        for charges in keys(core.Blocks)
            left_charge = charges[1]
            haskey(table, left_charge) && continue
            values = Vector{Y}(undef, length(domain))
            for (domain_index, x) in enumerate(domain)
                values[domain_index] = getval(lookup_buffer, core, left_charge, x)
            end
            table[left_charge] = values
        end
        tables[core_index] = table
        fallbacks[core_index] = zeros(Y, length(domain))
    end
    return tables, fallbacks
end

@inline function _weighted_index(rng::AbstractRNG, weights::Vector{Y}, count::Int) where {Y<:AbstractFloat}
    total = zero(Y)
    @inbounds for index in 1:count
        total += weights[index]
    end
    total > zero(Y) || throw(ArgumentError("sampling weights must have positive sum"))
    threshold = rand(rng, Y) * total
    cumulative = zero(Y)
    @inbounds for index in 1:count
        cumulative += weights[index]
        threshold < cumulative && return index
    end
    return count
end

function _sample_nondeg_column!(rng::AbstractRNG, mps::U1MPS{S, Y},
                                sampling_tables::Vector{Dict{Vector{S},Vector{Y}}},
                                fallbacks::Vector{Vector{Y}},
                                mem_buf::Matrix{S}, sample_iter::Int,
                                probability_vec::Vector{Y},
                                charge::Vector{S}) where {S<:Integer, Y<:AbstractFloat}
    core1 = mps.Cores[1]
    b_charge = first(core1.Indices[1].Charges)
    x1_domain = core1.Indices[2].FlatDomain
    copyto!(charge, b_charge)
    values = get(sampling_tables[1], charge, fallbacks[1])

    for (i, x1) in enumerate(x1_domain)
        probability_vec[i] = values[i] * values[i]
    end

    x1_ind = _weighted_index(rng, probability_vec, length(x1_domain))
    x1_val = x1_domain[x1_ind]
    mem_buf[1, sample_iter] = x1_val
    v = values[x1_ind]
    prob = probability_vec[x1_ind]
    @inbounds for constraint in eachindex(charge)
        charge[constraint] -= core1.Indices[2].A_vec[constraint] * x1_val
    end

    for j in 2:length(mps.Cores)
        core = mps.Cores[j]
        xj_domain = core.Indices[2].FlatDomain
        values = get(sampling_tables[j], charge, fallbacks[j])
        inv_sqrt_prob = 1 / sqrt(prob)
        for (i, xj) in enumerate(xj_domain)
            temp_v = inv_sqrt_prob * v * values[i]
            probability_vec[i] = temp_v * temp_v
        end
        xj_ind = _weighted_index(rng, probability_vec, length(xj_domain))
        xj_val = xj_domain[xj_ind]
        mem_buf[j, sample_iter] = xj_val
        v = inv_sqrt_prob * v * values[xj_ind]
        prob = probability_vec[xj_ind]
        @inbounds for constraint in eachindex(charge)
            charge[constraint] -= core.Indices[2].A_vec[constraint] * xj_val
        end
    end
    return nothing
end

"""
    sample_nondeg_parallel!(rng, mps, mem_buf, num_samples)

Sample feasible points from a non-degenerate U1-symmetric MPS in parallel.
Fixed logical blocks receive seeds in order before threading, so the result is
independent of thread scheduling and of the number of worker threads without
constructing a separate RNG for every sample.
"""
function sample_nondeg_parallel!(rng::AbstractRNG, mps::U1MPS{S, Y},
                                 mem_buf::Matrix{S}, num_samples::Int) where {S<:Integer, Y<:AbstractFloat}
    num_samples <= size(mem_buf, 2) || throw(ArgumentError("mem_buf has too few columns"))
    num_blocks = cld(num_samples, _RNG_BLOCK_SIZE)
    block_seeds = rand(rng, UInt64, num_blocks)
    max_domain_size = maximum(length(c.Indices[2].FlatDomain) for c in mps.Cores)
    num_constraints = length(first(mps.Cores[1].Indices[1].Charges))
    num_thread_ids = Threads.maxthreadid()
    prob_vec_mem = [zeros(Y, max_domain_size) for _ in 1:num_thread_ids]
    charge_mem = [Vector{S}(undef, num_constraints) for _ in 1:num_thread_ids]
    sampling_tables, fallbacks = _build_nondeg_sampling_tables(mps)

    Threads.@threads for block in 1:num_blocks
        tid = Threads.threadid()
        block_rng = Random.Xoshiro(block_seeds[block])
        first_sample = (block - 1) * _RNG_BLOCK_SIZE + 1
        last_sample = min(block * _RNG_BLOCK_SIZE, num_samples)
        for sample_iter in first_sample:last_sample
            _sample_nondeg_column!(block_rng, mps, sampling_tables, fallbacks,
                                  mem_buf, sample_iter, prob_vec_mem[tid], charge_mem[tid])
        end
    end
    return nothing
end

sample_nondeg_parallel!(mps::U1MPS, mem_buf::Matrix, num_samples::Int) =
    sample_nondeg_parallel!(Random.default_rng(), mps, mem_buf, num_samples)

"""
    sample_nondeg!(rng, mps, mem_buf, num_samples)

Serial counterpart of `sample_nondeg_parallel!`. It intentionally uses the
same logical-block seed scheme, so serial and parallel calls are identical for the
same initial RNG state.
"""
function sample_nondeg!(rng::AbstractRNG, mps::U1MPS{S, Y},
                        mem_buf::Matrix{S}, num_samples::Int) where {S<:Integer, Y<:AbstractFloat}
    num_samples <= size(mem_buf, 2) || throw(ArgumentError("mem_buf has too few columns"))
    num_blocks = cld(num_samples, _RNG_BLOCK_SIZE)
    block_seeds = rand(rng, UInt64, num_blocks)
    max_domain_size = maximum(length(c.Indices[2].FlatDomain) for c in mps.Cores)
    num_constraints = length(first(mps.Cores[1].Indices[1].Charges))
    probability_vec = zeros(Y, max_domain_size)
    charge = Vector{S}(undef, num_constraints)
    sampling_tables, fallbacks = _build_nondeg_sampling_tables(mps)

    for block in 1:num_blocks
        block_rng = Random.Xoshiro(block_seeds[block])
        first_sample = (block - 1) * _RNG_BLOCK_SIZE + 1
        last_sample = min(block * _RNG_BLOCK_SIZE, num_samples)
        for sample_iter in first_sample:last_sample
            _sample_nondeg_column!(block_rng, mps, sampling_tables, fallbacks,
                                  mem_buf, sample_iter, probability_vec, charge)
        end
    end
    return nothing
end

sample_nondeg!(mps::U1MPS, mem_buf::Matrix, num_samples::Int) =
    sample_nondeg!(Random.default_rng(), mps, mem_buf, num_samples)
