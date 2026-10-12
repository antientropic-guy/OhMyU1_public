# collect_left/right cache actual positions, including absent physical sectors
# and zero columns. Keep the site argument for existing callers.
@inline function nondeg_merged_index(matrix, charge, site, x)
    position = get(matrix.Xpositions, (charge, x), 0)
    iszero(position) && error("Training sample is outside the MPS support")
    return position
end

# Normalized cross entropy. The MPS must be in mixed canonical form.
function Loss(mps::U1MPS, x_samples::Matrix{S}, probs) where {S<:Integer}
    loss = sum(probs) * log(u1_norm(mps)^2)
    for (i, x) in enumerate(eachcol(x_samples))
        iszero(probs[i]) || (loss -= probs[i] * log(mps[x] ^2))
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

function sweep_right_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams; weight_sum=sum(probs)) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
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
        multiply_factor = (1 - 2 * alpha * weight_sum / z_i_norm)
        for c in keys(merged_cores.Blocks)
            # merged_cores.Blocks[c] .-= alpha * merged_cores.Blocks[c] ./ z_i_norm
            merged_cores.Blocks[c] .*= multiply_factor
        end

        # Part 2: -2P(x)MPS'(x)/MPS(x)
        for (p, x) in enumerate(eachcol(x_samples))
            iszero(probs[p]) && continue
            
            left_charge_mem_buf .= @view link_charges[i, :, p]
            mid_charge_mem_buf .= @view link_charges[i+1, :, p]
            right_charge_mem_buf .= @view link_charges[i+2, :, p] 

            i1 = nondeg_merged_index(l_matrix, mid_charge_mem_buf, l_core.Indices[2], x[i])
            j1 = nondeg_merged_index(r_matrix, mid_charge_mem_buf, r_core.Indices[2], x[i+1])

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

function sweep_left_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams; weight_sum=sum(probs)) where {S<:Integer, Y<:AbstractFloat}
    b = first(mps.Cores[1].Indices[1].Charges)
    N_constr = length(b)
    d = length(mps.Cores)
    A = mps.A
    link_charges = compute_link_charges(A, b, x_samples)
    alpha = train_params.LearningRate

    # Memory allocation:
    left_charge_mem_buf = Vector{S}(undef, N_constr)
    mid_charge_mem_buf = Vector{S}(undef, N_constr)
    right_charge_mem_buf = Vector{S}(undef, N_constr)

    # right -> left sweep:
    for i in reverse(1:d-1)
        @assert mps.ort_center == i + 1
        l_core = mps.Cores[i]
        r_core = mps.Cores[i+1]
        merged_cores = MergedCores(l_core, r_core)
        l_matrix = merged_cores.left_matrix
        r_matrix = merged_cores.right_matrix
        z_i_norm = u1_norm(mps) ^ 2

        # Part 1: Z'|Z:
        multiply_factor = (1 - 2 * alpha * weight_sum / z_i_norm)
        for c in keys(merged_cores.Blocks)
            # merged_cores.Blocks[c] .-= 2 alpha * merged_cores.Blocks[c] ./ z_i_norm
            merged_cores.Blocks[c] .*= multiply_factor
        end

        # Part 2: -2P(x)MPS'(x)/MPS(x)

        for (p, x) in enumerate(eachcol(x_samples))
            iszero(probs[p]) && continue
            left_charge_mem_buf .= @view link_charges[i, :, p]
            mid_charge_mem_buf .= @view link_charges[i+1, :, p]
            right_charge_mem_buf .= @view link_charges[i+2, :, p] 

            i1 = nondeg_merged_index(l_matrix, mid_charge_mem_buf, l_core.Indices[2], x[i])
            j1 = nondeg_merged_index(r_matrix, mid_charge_mem_buf, r_core.Indices[2], x[i+1])

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
    @assert mps.ort_center == 1 "Canonicalize at site 1 before training"
    @assert length(probs) == size(x_samples, 2)
    weight_sum = sum(probs)
    @assert all(isfinite, probs) && all(>=(0), probs) && weight_sum > 0
    @assert all(size(B,1)==size(B,3)==1 for c in mps.Cores for B in values(c.Blocks)) "Nondegenerate training requires sector dimensions equal to one"
    for i in 1:num_iter
        sweep_right_nondeg!(mps, x_samples, probs, train_params; weight_sum)
        sweep_left_nondeg!(mps, x_samples, probs, train_params; weight_sum)
        # push!(loss_arr, Loss(mps, x_samples, probs))
    end
    # return loss_arr
end