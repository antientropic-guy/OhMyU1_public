# Sample feasible points from U1-symmetric MPS:
function sample(mps::U1MPS{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    
    sampled_x = Vector{S}()
    max_domain_size = maximum([length(c.Indices[2].FlatDomain) for c in mps.Cores])
    
    core1 = mps.Cores[1]
    b_charge = first(core1.Indices[1].Charges)
    N_constr = length(b_charge)
    
    x1_domain = core1.Indices[2].FlatDomain
    v_vec = Vector{Matrix{Y}}(undef, max_domain_size)
    probability_vec = Vector{Y}(undef, max_domain_size)
    for (i, x1) in enumerate(x1_domain)
        v_vec[i] = core1[b_charge, x1]
        probability_vec[i] = sum(abs2, v_vec[i])
    end
    x1_ind = StatsBase.sample(1:length(x1_domain), Weights(probability_vec))

    # Input to the cycle:
    x1_val = x1_domain[x1_ind]
    push!(sampled_x, x1_val)
    v = v_vec[x1_ind]
    prob = probability_vec[x1_ind]

    #TODO: simplify logic
    site_charge, _ = core1.Indices[2].InvXindex[x1_val]
    left_charge = b_charge .- site_charge

    for j in 2:length(mps.Cores)
        core = mps.Cores[j]
        # TODO: take out of this cycle;
        xj_domain = core.Indices[2].FlatDomain
        for (i, xj) in enumerate(xj_domain)
            temp_v = 1 / sqrt(prob) * v * core[left_charge, xj]
            v_vec[i] = temp_v
            probability_vec[i] = sum(abs2, temp_v)
        end
        xj_ind = StatsBase.sample(1:length(xj_domain), Weights(probability_vec))

        # Input to the cycle:
        xj_val = xj_domain[xj_ind]
        push!(sampled_x, xj_val)
        v = v_vec[xj_ind]
        prob = probability_vec[xj_ind]
    
        #TODO: simplify logic
        site_charge, _ = core.Indices[2].InvXindex[xj_val]
        left_charge .-= site_charge

    end
    return sampled_x
end

# Sample feasible points from U1-symmetric MPS:
function sample_nondeg(mps::U1MPS{S, Y}) where {S<:Integer, Y<:AbstractFloat}
    
    sampled_x = Vector{S}()
    max_domain_size = maximum([length(c.Indices[2].FlatDomain) for c in mps.Cores])
    
    core1 = mps.Cores[1]
    b_charge = first(core1.Indices[1].Charges)
    N_constr = length(b_charge)
    
    x1_domain = core1.Indices[2].FlatDomain
    v_vec = Vector{Y}(undef, max_domain_size)
    probability_vec = Vector{Y}(undef, max_domain_size)
    for (i, x1) in enumerate(x1_domain)
        v_vec[i] = getval(core1, b_charge, x1)
        probability_vec[i] = v_vec[i] * v_vec[i]
    end
    x1_ind = StatsBase.sample(1:length(x1_domain), Weights(probability_vec))

    # Input to the cycle:
    x1_val = x1_domain[x1_ind]
    push!(sampled_x, x1_val)
    v = v_vec[x1_ind]
    prob = probability_vec[x1_ind]

    #TODO: simplify logic
    site_charge, _ = core1.Indices[2].InvXindex[x1_val]
    left_charge = b_charge .- site_charge

    for j in 2:length(mps.Cores)
        core = mps.Cores[j]
        # TODO: take out of this cycle;
        xj_domain = core.Indices[2].FlatDomain
        inv_sqrt_prob = 1 / sqrt(prob)
        for (i, xj) in enumerate(xj_domain)
            temp_v = inv_sqrt_prob * v * getval(core, left_charge, xj)
            v_vec[i] = temp_v
            probability_vec[i] = temp_v * temp_v
        end
        xj_ind = StatsBase.sample(1:length(xj_domain), Weights(probability_vec))

        # Input to the cycle:
        xj_val = xj_domain[xj_ind]
        push!(sampled_x, xj_val)
        v = v_vec[xj_ind]
        prob = probability_vec[xj_ind]
    
        #TODO: simplify logic
        site_charge, _ = core.Indices[2].InvXindex[xj_val]
        left_charge .-= site_charge

    end
    return sampled_x
end

# Sample feasible points from U1-symmetric MPS:
function sample_nondeg!(mps::U1MPS{S, Y}, mem_buf::Matrix{S}, num_samples::Int) where {S<:Integer, Y<:AbstractFloat}

    max_domain_size = maximum([length(c.Indices[2].FlatDomain) for c in mps.Cores])
    core1 = mps.Cores[1]
    b_charge = first(core1.Indices[1].Charges)
    N_constr = length(b_charge)
    x1_domain = core1.Indices[2].FlatDomain
    v_vec = Vector{Y}(undef, max_domain_size)
    probability_vec = Vector{Y}(undef, max_domain_size)
    
    for sample_iter in 1:num_samples
        
        for (i, x1) in enumerate(x1_domain)
            v_vec[i] = getval(core1, b_charge, x1)
            probability_vec[i] = v_vec[i] * v_vec[i]
        end
        x1_ind = StatsBase.sample(1:length(x1_domain), Weights(probability_vec))
    
        # Input to the cycle:
        x1_val = x1_domain[x1_ind]
        mem_buf[1, sample_iter] = x1_val
        v = v_vec[x1_ind]
        prob = probability_vec[x1_ind]
    
        #TODO: simplify logic
        site_charge, _ = core1.Indices[2].InvXindex[x1_val]
        left_charge = b_charge .- site_charge
    
        for j in 2:length(mps.Cores)
            core = mps.Cores[j]
            xj_domain = core.Indices[2].FlatDomain
            inv_sqrt_prob = 1 / sqrt(prob)
            for (i, xj) in enumerate(xj_domain)
                temp_v = inv_sqrt_prob * v * getval(core, left_charge, xj)
                v_vec[i] = temp_v
                probability_vec[i] = temp_v * temp_v
            end
            xj_ind = StatsBase.sample(1:length(xj_domain), Weights(probability_vec))
    
            # Input to the cycle:
            xj_val = xj_domain[xj_ind]
            mem_buf[j, sample_iter] = xj_val
            v = v_vec[xj_ind]
            prob = probability_vec[xj_ind]
        
            #TODO: simplify logic
            site_charge, _ = core.Indices[2].InvXindex[xj_val]
            left_charge .-= site_charge
    
        end
    end
    return nothing
end