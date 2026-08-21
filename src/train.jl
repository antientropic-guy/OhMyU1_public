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

# Sweeps (for non-degenerate links case only!)

function sweep_right_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
    zero_charge = zeros(S, N_constr)
    link_charges = compute_link_charges(A, b, x_samples)
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
        for (p, x) in enumerate(eachcol(x_samples))
            
            left_charge_mem_buf .= @view link_charges[i, :, p]
            mid_charge_mem_buf .= @view link_charges[i+1, :, p]
            right_charge_mem_buf .= @view link_charges[i+2, :, p] 

            _, x_ind_l = l_core.Indices[2].InvXindex[x[i]]
            _, x_ind_r = r_core.Indices[2].InvXindex[x[i+1]]
            
            if any((@view A[:, i]) .!= 0)
                l_x_bounds = l_matrix.XboundsDict[mid_charge_mem_buf]
                i1 = l_x_bounds[x_ind_l] + 1
            else
                i1 = left_shape * (x_ind_l - 1) + 1
            end
    
            if any((@view A[:, i+1]) .!= 0)
                r_x_bounds = r_matrix.XboundsDict[mid_charge_mem_buf]
                j1 = r_x_bounds[x_ind_r] + 1
            else
                j1 = right_shape * (x_ind_r - 1) + 1
            end

            merged_cores.Blocks[mid_charge_mem_buf][i1, j1] += 2 * alpha * probs[p] / getval(l_core, left_charge_mem_buf, x[i]) / getval(r_core, x[i+1], right_charge_mem_buf)
        end

        for charge in keys(merged_cores.Blocks)
            block = merged_cores.Blocks[charge]
            U, Sigma, V = svd(block)
            l_matrix.Blocks[charge] = U[:, 1:1]
            new_V = Diagonal(Sigma) * V'
            r_matrix.Blocks[charge] = new_V[1:1, :]
        end
        mps.ort_center = i + 1
        mps.Cores[i] = split_left(l_matrix)
        mps.Cores[i+1] = split_right(r_matrix)
    end
end

function sweep_left_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
    # zero_charge = zeros(S, N_constr)
    link_charges = compute_link_charges(A, b, x_samples)
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
            U, Sigma, V = svd(block)
            l_matrix.Blocks[charge] = U[:, 1:1] .* Sigma[1]
            r_matrix.Blocks[charge] = V'[1:1, :]
        end
        mps.ort_center = i
        mps.Cores[i] = split_left(l_matrix)
        mps.Cores[i+1] = split_right(r_matrix)
    end
end

# Learn probability distribution (minimizing cross-entropy loss) - for non-degenerate links case;
function train_nondeg!(mps, num_iter, x_samples, probs, train_params)
    # loss_arr = [Loss(mps, x_samples, probs)]
    for i in 1:num_iter
        sweep_right_nondeg!(mps, x_samples, probs, train_params)
        sweep_left_nondeg!(mps, x_samples, probs, train_params)
        # push!(loss_arr, Loss(mps, x_samples, probs))
    end
    # return loss_arr
end