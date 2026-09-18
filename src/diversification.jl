# ============================================================================
# File: diversification.jl
# Part of: OhMyU1
#
# Description:
#   Helper-functions for diversification of charge graphs
#
# Author: Sergei
# Created: Feb 2026
# ============================================================================


function update_index!(link_index::ChargeIndex{T}, charge::Vector{T}) where T <: Integer
    push!(link_index.Charges, charge)
    link_index.Card += 1
end


"""
    diversify!(mps::U1MPS{S, N}) where {S<:Integer, N<:AbstractFloat} -> Nothing
Adds new charges to the MPS to diversify the charge graph.
# Arguments
- `mps::U1MPS{S, N}`: U1MPS object to diversify
# Returns
- `Nothing`: The function modifies the MPS in place.
"""
function diversify!(mps::U1MPS{S, N}) where {S<:Integer, N<:AbstractFloat}
    N_sites = length(mps.Cores)
    diversification_counter = 0
    for i in 3:N_sites-1
        left_core = mps.Cores[i-1]
        right_core = mps.Cores[i]
        left_link_index = left_core.Indices[1]
        right_link_index = right_core.Indices[3]
        mid_link_index = left_core.Indices[3]
        left_site_ind = left_core.Indices[2]
        right_site_ind = right_core.Indices[2] 
        
        site_deg_l = 1
        site_card_l = length(left_site_ind.FlatDomain)
        if site_card_l > length(left_site_ind.Charges)
            site_deg_l = site_card_l
        end

        site_deg_r = 1
        site_card_r = length(right_site_ind.FlatDomain)
        if site_card_r > length(right_site_ind.Charges)
            site_deg_r = site_card_r
        end



        for left_link_charge in left_link_index.Charges
            for left_x in left_site_ind.FlatDomain
                for right_x in right_site_ind.FlatDomain
                    left_site_charge = left_site_ind.A_vec .* left_x
                    right_site_charge = right_site_ind.A_vec .* right_x
                    charge_vec_r = left_link_charge .- left_site_charge .- right_site_charge
                    if in(charge_vec_r, right_link_index.Charges)
                        new_mid_charge = charge_vec_r .+ right_site_charge
                        if !in(new_mid_charge, mid_link_index.Charges)
                            diversification_counter += 1
                            update_index!(mid_link_index, new_mid_charge)
                            right_core.Blocks[(new_mid_charge, right_site_charge, charge_vec_r)] = ones(1, site_deg_r, 1)
                            left_core.Blocks[(left_link_charge, left_site_charge, new_mid_charge)] = ones(1, site_deg_l, 1)
                        end
                    end
                end
            end
        end
    end
    println("Added during diversification: ", diversification_counter)
    return nothing
end


"""
    graph_dist(sample, mps::U1MPS{T, N}) where {T<:Integer, N<:AbstractFloat} -> Int
Computes the graph distance of a sample from the MPS charge graph.
# Arguments
- `sample`: Sample vector to evaluate   
- `mps::U1MPS{T, N}`: U1MPS object
# Returns 
- `Int`: Graph distance of the sample from the MPS charge graph.
"""
function graph_dist(sample, mps::U1MPS{T, N}) where {T<:Integer, N<:AbstractFloat}
    charge_matrix = ein"ij, j -> ij"(-mps.A, sample)
    charge_matrix = cumsum(charge_matrix; dims=2)
    charge_matrix .= charge_matrix .+ first(mps.LinkIndices[1].Charges)
    dist = 0
    for i in 1:size(charge_matrix, 2)
        if !(charge_matrix[:, i] in mps.LinkIndices[i + 1].Charges)
            dist += 1
        end
    end

    return dist
end

