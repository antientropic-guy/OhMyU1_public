# Cross-entropy loss:
function Loss(mps::U1MPS, x_samples::Matrix{S}, probs) where {S<:Integer}
    loss = 0
    for (i, x) in enumerate(eachcol(x_samples))
        loss -= probs[i] * log(mps[x] ^2)
    end
    return loss
end

# Some train params (with some not used currently fields)
struct TrainParams
    LearningRate::AbstractFloat
    MaxR::Int
    CutOff::AbstractFloat
end

TrainParams(learning_rate::AbstractFloat, max_r::Int) = TrainParams(learning_rate, max_r, 1e-12) 

"""Leading singular triplet with allocation-light formulas for the tiny blocks used by rank-one MPS."""
function _dominant_singular_triplet(block::AbstractMatrix{Y}) where {Y<:AbstractFloat}
    m, n = size(block)
    if m == 1
        sigma = norm(block)
        iszero(sigma) && return Y[one(Y)], zero(Y), vcat(one(Y), zeros(Y, n - 1))
        return Y[one(Y)], sigma, vec(copy(block)) ./ sigma
    elseif n == 1
        sigma = norm(block)
        iszero(sigma) && return vcat(one(Y), zeros(Y, m - 1)), zero(Y), Y[one(Y)]
        return vec(copy(block)) ./ sigma, sigma, Y[one(Y)]
    elseif m == 2 && n == 2
        a11, a21 = block[1, 1], block[2, 1]
        a12, a22 = block[1, 2], block[2, 2]
        gram11 = abs2(a11) + abs2(a21)
        gram12 = a11 * a12 + a21 * a22
        gram22 = abs2(a12) + abs2(a22)
        lambda = (gram11 + gram22 + hypot(gram11 - gram22, 2gram12)) / 2
        sigma = sqrt(max(lambda, zero(Y)))
        iszero(sigma) && return Y[one(Y), zero(Y)], zero(Y), Y[one(Y), zero(Y)]

        candidate1 = (gram12, lambda - gram11)
        candidate2 = (lambda - gram22, gram12)
        v1, v2 = sum(abs2, candidate1) >= sum(abs2, candidate2) ? candidate1 : candidate2
        vnorm = hypot(v1, v2)
        if iszero(vnorm)
            v1, v2 = gram11 >= gram22 ? (one(Y), zero(Y)) : (zero(Y), one(Y))
        else
            v1 /= vnorm
            v2 /= vnorm
        end
        u1 = (a11 * v1 + a12 * v2) / sigma
        u2 = (a21 * v1 + a22 * v2) / sigma
        return Y[u1, u2], sigma, Y[v1, v2]
    else
        factorization = svd(block; full=false)
        return Vector(factorization.U[:, 1]), factorization.S[1], Vector(factorization.V[:, 1])
    end
end

# Sweeps (for non-degenerate links case only!)

function sweep_right_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y},
                             train_params::TrainParams,
                             link_charges=compute_link_charges(mps.A, first(mps.Cores[1].Indices[1].Charges), x_samples)) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
    zero_charge = zeros(S, N_constr)
    alpha = train_params.LearningRate

    # Memory allocation:
    left_charge_mem_buf = Vector{S}(undef, N_constr)
    mid_charge_mem_buf = Vector{S}(undef, N_constr)
    right_charge_mem_buf = Vector{S}(undef, N_constr)

    # left -> right sweep:
    for i in 1:d-1
        @assert mps.ort_center == i
        l_core = mps.Cores[i]
        r_core = mps.Cores[i+1]
        merged_cores = MergedCores(l_core, r_core)
        l_matrix = merged_cores.left_matrix
        r_matrix = merged_cores.right_matrix
        z_i_norm = u1_norm(mps) ^ 2

        # Part 1: Z'|Z:
        multiply_factor = (1 - 2 * alpha / z_i_norm)
        for c in keys(merged_cores.Blocks)
            # merged_cores.Blocks[c] .-= alpha * merged_cores.Blocks[c] ./ z_i_norm
            merged_cores.Blocks[c] .*= multiply_factor
        end

        # Part 2: -2P(x)MPS'(x)/MPS(x)
        statement_i = any(!iszero, @view A[:, i])
        statement_i_plus_1 = any(!iszero, @view A[:, i+1])
        for (p, x) in enumerate(eachcol(x_samples))
            
            left_charge_mem_buf .= @view link_charges[i, :, p]
            mid_charge_mem_buf .= @view link_charges[i+1, :, p]
            right_charge_mem_buf .= @view link_charges[i+2, :, p] 

            _, x_ind_l = l_core.Indices[2].InvXindex[x[i]]
            _, x_ind_r = r_core.Indices[2].InvXindex[x[i+1]]
            
            if statement_i
                l_x_bounds = l_matrix.XboundsDict[mid_charge_mem_buf]
                i1 = l_x_bounds[x_ind_l] + 1
            else
                i1 = left_shape * (x_ind_l - 1) + 1
            end
    
            if statement_i_plus_1
                r_x_bounds = r_matrix.XboundsDict[mid_charge_mem_buf]
                j1 = r_x_bounds[x_ind_r] + 1
            else
                j1 = right_shape * (x_ind_r - 1) + 1
            end

            merged_cores.Blocks[mid_charge_mem_buf][i1, j1] += 2 * alpha * probs[p] / getval(l_core, left_charge_mem_buf, x[i]) / getval(r_core, x[i+1], right_charge_mem_buf)
        end

        for charge in keys(merged_cores.Blocks)
            block = merged_cores.Blocks[charge]
            u, sigma, v = _dominant_singular_triplet(block)
            l_matrix.Blocks[charge] = reshape(u, :, 1)
            r_matrix.Blocks[charge] = reshape(sigma .* v, 1, :)
        end
        mps.ort_center = i + 1
        mps.Cores[i] = split_left(l_matrix)
        mps.Cores[i+1] = split_right(r_matrix)
    end
end

function sweep_left_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y},
                            train_params::TrainParams,
                            link_charges=compute_link_charges(mps.A, first(mps.Cores[1].Indices[1].Charges), x_samples)) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
    # zero_charge = zeros(S, N_constr)
    alpha = train_params.LearningRate

    # Memory allocation:
    left_charge_mem_buf = Vector{S}(undef, N_constr)
    mid_charge_mem_buf = Vector{S}(undef, N_constr)
    right_charge_mem_buf = Vector{S}(undef, N_constr)

    # left -> right sweep:
    for i in reverse(1:d-1)
        @assert mps.ort_center == i + 1
        l_core = mps.Cores[i]
        r_core = mps.Cores[i+1]
        merged_cores = MergedCores(l_core, r_core)
        l_matrix = merged_cores.left_matrix
        r_matrix = merged_cores.right_matrix
        z_i_norm = u1_norm(mps) ^ 2

        # Part 1: Z'|Z:
        multiply_factor = (1 - 2 * alpha / z_i_norm)
        for c in keys(merged_cores.Blocks)
            # merged_cores.Blocks[c] .-= 2 alpha * merged_cores.Blocks[c] ./ z_i_norm
            merged_cores.Blocks[c] .*= multiply_factor
        end

        # Part 2: -2P(x)MPS'(x)/MPS(x)
        statement_i = any(!iszero, @view A[:, i])
        statement_i_plus_1 = any(!iszero, @view A[:, i+1])

        for (p, x) in enumerate(eachcol(x_samples))
            left_charge_mem_buf .= @view link_charges[i, :, p]
            mid_charge_mem_buf .= @view link_charges[i+1, :, p]
            right_charge_mem_buf .= @view link_charges[i+2, :, p] 

            _, x_ind_l = l_core.Indices[2].InvXindex[x[i]]
            _, x_ind_r = r_core.Indices[2].InvXindex[x[i+1]]
            
            if statement_i
                l_x_bounds = l_matrix.XboundsDict[mid_charge_mem_buf]
                i1 = l_x_bounds[x_ind_l] + 1
            else
                i1 = left_shape * (x_ind_l - 1) + 1
            end
    
            if statement_i_plus_1
                r_x_bounds = r_matrix.XboundsDict[mid_charge_mem_buf]
                j1 = r_x_bounds[x_ind_r] + 1
            else
                j1 = right_shape * (x_ind_r - 1) + 1
            end

            merged_cores.Blocks[mid_charge_mem_buf][i1, j1] += 2 * alpha * probs[p] / getval(l_core, left_charge_mem_buf, x[i]) / getval(r_core, x[i+1], right_charge_mem_buf)
        end

        for charge in keys(merged_cores.Blocks)
            block = merged_cores.Blocks[charge]
            u, sigma, v = _dominant_singular_triplet(block)
            l_matrix.Blocks[charge] = reshape(sigma .* u, :, 1)
            r_matrix.Blocks[charge] = reshape(v, 1, :)
        end
        mps.ort_center = i
        mps.Cores[i] = split_left(l_matrix)
        mps.Cores[i+1] = split_right(r_matrix)
    end
end

# Learn probability distribution (minimizing cross-entropy loss) - for non-degenerate links case;
function train_nondeg!(mps, num_iter, x_samples, probs, train_params)
    # loss_arr = [Loss(mps, x_samples, probs)]
    link_charges = compute_link_charges(
        mps.A, first(mps.Cores[1].Indices[1].Charges), x_samples)
    for i in 1:num_iter
        sweep_right_nondeg!(mps, x_samples, probs, train_params, link_charges)
        sweep_left_nondeg!(mps, x_samples, probs, train_params, link_charges)
        # push!(loss_arr, Loss(mps, x_samples, probs))
    end
    # return loss_arr
end
