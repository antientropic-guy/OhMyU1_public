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
struct TrainParams{T<:AbstractFloat,C<:AbstractFloat}
    LearningRate::T
    MaxR::Int
    CutOff::C
    Backend::Symbol
end

function TrainParams(learning_rate::AbstractFloat, max_r::Int, cutoff::AbstractFloat=1e-12; backend::Symbol=:fast)
    backend in (:fast, :lapack) || throw(ArgumentError("Unknown factorization backend: $backend"))
    TrainParams(learning_rate, max_r, cutoff, backend)
end

# Only the dominant triplet is needed for nondegenerate charge sectors.
# Scale first: squaring unscaled entries can overflow or underflow.
function dominant_triplet(B::AbstractMatrix{T}, backend::Symbol=:fast) where {T<:AbstractFloat}
    rows, cols = size(B)
    if backend == :lapack || !(rows == 1 || cols == 1 || (rows == cols == 2))
        F = svd(B)
        return F.U[:, 1], F.S[1], F.V[:, 1]
    end
    scale = maximum(abs, B)
    u = zeros(T, rows); v = zeros(T, cols)
    if iszero(scale)
        u[1] = v[1] = one(T)
        return u, zero(T), v
    elseif rows == 1
        v .= vec(B) ./ scale
        s = norm(v); v ./= s; u[1] = one(T)
        return u, scale * s, v
    elseif cols == 1
        u .= vec(B) ./ scale
        s = norm(u); u ./= s; v[1] = one(T)
        return u, scale * s, v
    end
    a, b = B[1,1]/scale, B[1,2]/scale
    c, d = B[2,1]/scale, B[2,2]/scale
    g11, g22, g12 = a*a+c*c, b*b+d*d, a*b+c*d
    lambda = (g11 + g22 + hypot(g11-g22, 2g12)) / 2
    x, y = g12, lambda-g11
    if hypot(lambda-g22, g12) > hypot(x,y)
        x, y = lambda-g22, g12
    end
    h = hypot(x,y)
    if iszero(h)
        v[1] = one(T)
    else
        v[1], v[2] = x/h, y/h
    end
    u[1], u[2] = a*v[1]+b*v[2], c*v[1]+d*v[2]
    s = norm(u); u ./= s
    return u, scale*s, v
end

# Samples with the same middle charge and physical pair contribute the same
# local log-amplitude derivative. Sum their target weights once per train call.
function pair_training_weights(x_samples::Matrix{S}, probs::Vector{Y}, link_charges) where {S,Y}
    targets = [Dict{Tuple{Vector{S},S,S},Y}() for _ in 1:size(x_samples,1)-1]
    for i in eachindex(targets), p in axes(x_samples,2)
        iszero(probs[p]) && continue
        key = (link_charges[i+1,:,p],x_samples[i,p],x_samples[i+1,p])
        targets[i][key] = get(targets[i],key,zero(Y)) + probs[p]
    end
    targets
end

# Sweeps (for non-degenerate links case only!)

function sweep_right_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams; weight_sum=sum(probs), link_charges=compute_link_charges(mps.A, first(mps.LinkIndices[1].Charges), x_samples), targets=pair_training_weights(x_samples,probs,link_charges)) where {S<:Integer, Y<:AbstractFloat}
    d = length(mps.Cores)
    alpha = train_params.LearningRate

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
        for ((charge, x, y), weight) in targets[i]
            row = nondeg_merged_index(l_matrix, charge, l_core.Indices[2], x)
            col = nondeg_merged_index(r_matrix, charge, r_core.Indices[2], y)
            left_value = l_matrix.Blocks[charge][row,1]
            right_value = r_matrix.Blocks[charge][1,col]
            merged_cores.Blocks[charge][row,col] += 2 * alpha * weight / left_value / right_value
        end

        for charge in keys(merged_cores.Blocks)
            block = merged_cores.Blocks[charge]
            u, sigma, v = dominant_triplet(block, train_params.Backend)
            l_matrix.Blocks[charge] = reshape(u, :, 1)
            r_matrix.Blocks[charge] = reshape(v .* sigma, 1, :)
        end
        mps.ort_center = i + 1
        mps.Cores[i] = split_left(l_matrix)
        mps.Cores[i+1] = split_right(r_matrix)
    end
end

function sweep_left_nondeg!(mps::U1MPS{S, Y}, x_samples::Matrix{S}, probs::Vector{Y}, train_params::TrainParams; weight_sum=sum(probs), link_charges=compute_link_charges(mps.A, first(mps.LinkIndices[1].Charges), x_samples), targets=pair_training_weights(x_samples,probs,link_charges)) where {S<:Integer, Y<:AbstractFloat}
    d = length(mps.Cores)
    alpha = train_params.LearningRate

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

        for ((charge, x, y), weight) in targets[i]
            row = nondeg_merged_index(l_matrix, charge, l_core.Indices[2], x)
            col = nondeg_merged_index(r_matrix, charge, r_core.Indices[2], y)
            left_value = l_matrix.Blocks[charge][row,1]
            right_value = r_matrix.Blocks[charge][1,col]
            merged_cores.Blocks[charge][row,col] += 2 * alpha * weight / left_value / right_value
        end

        for charge in keys(merged_cores.Blocks)
            block = merged_cores.Blocks[charge]
            u, sigma, v = dominant_triplet(block, train_params.Backend)
            l_matrix.Blocks[charge] = reshape(u .* sigma, :, 1)
            r_matrix.Blocks[charge] = reshape(v, 1, :)
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
    link_charges = compute_link_charges(mps.A, first(mps.LinkIndices[1].Charges), x_samples)
    targets = pair_training_weights(x_samples,probs,link_charges)
    for i in 1:num_iter
        sweep_right_nondeg!(mps, x_samples, probs, train_params; weight_sum, link_charges, targets)
        sweep_left_nondeg!(mps, x_samples, probs, train_params; weight_sum, link_charges, targets)
        # push!(loss_arr, Loss(mps, x_samples, probs))
    end
    # return loss_arr
end
