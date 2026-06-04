Base.@kwdef struct SolverParams
    NUM_FEASIBLE_SAMPLES::Int = 1000
    NUM_MPS_SAMPLES::Int = 10000
    LINK_DEGENERACY::Int = 1
    NUM_SWEEP_ITER::Int = 1
    NUM_GLOBAL_ITER::Int = 10
    LEARNING_RATE::Float64 = 0.05
    UTILITY_FRACTION::Float64 = 0.05
    KEEP_NUM_WORST::Float64 = 0.05
end

function get_mps(A, b, num_feasible_samples::Int)

    x_domains = [[0, 1] for _ in 1:size(A, 2)]
    LINK_DEGENERACY = 1
    site_cards = fill(2, size(A, 2))
    
    T = sample_feasible_lp(A, b, num_feasible_samples)
    link_charges = compute_link_charges(A, b, T)
    site_charges = compute_site_charges(A, x_domains)
    link_indices, site_indices = compute_indices(link_charges, site_charges, b, x_domains, A)
    mps = init_u1_mps(Float64, link_indices, site_indices, A, b, LINK_DEGENERACY, site_cards)
return mps
end

function graph_dist_global(sample::AbstractVector{T}, mps::U1MPS{T, N}, memorized_charges::Vector{Set{Vector{T}}}) where {T<:Integer, N<:AbstractFloat}
    dist = 0
    current_charge = copy(first(mps.LinkIndices[1].Charges))
    for i in 1:size(mps.A, 2)
        current_charge .+= (-view(mps.A, :, i) .* sample[i]) 
        if !(current_charge in memorized_charges[i])
            dist += 1
        end
    end

    return dist
end

function graph_dist_global!(buffer::Vector{T}, sample, mps, memorized_charges) where {T}
    dist = 0
    initial_charge = first(mps.LinkIndices[1].Charges)
    copyto!(buffer, initial_charge) 

    for i in 1:size(mps.A, 2)
        for k in 1:length(buffer)
            @inbounds buffer[k] -= mps.A[k, i] * sample[i]
        end
        
        if !(buffer in memorized_charges[i])
            dist += 1
        end
    end
    return dist
end


"""Represents optimization problem with linear equality constraints, integer variables and any black-box cost function. """
Base.@kwdef struct OptimizationProblem
    A::Matrix{Int}
    b::Vector{Int}
    cost_function::Function
    name::String = ""
    x_domains::Vector{Vector{Int}} = [[0, 1] for _ in 1:size(A, 2)]
    site_card::Vector{Int} = fill(2, size(A, 2))
    num_vars::Int = size(A, 2)
    num_constraints::Int = size(A, 1)
end

"""Represents solver statistics, which are updated during the optimization process and printed at each iteration."""
Base.@kwdef mutable struct SolverStatistics
    learning_curve::Vector{Real} = Real[]
    utility_curve::Vector{Real} = Real[]
    c_min::Real = Inf
    temperature::Real = Inf
    num_to_keep_worst::Int = -1
    num_samples_in_mps::Int = -1
    num_unique_samples::Int = -1
    num_principal_new::Int = -1
    max_graph_dist::Int = -1
    min_graph_dist::Int = -1
    incub::Vector{Int} = Int[]
end

"""Function prints statistics of the optimization process, which are stored in the `SolverStatistics` struct. It is called at each iteration to provide insights into the optimization progress."""
function print_statistics(stats::SolverStatistics)
    for name in fieldnames(SolverStatistics)
        value = getfield(stats, name)

        if value isa AbstractVector
            if isempty(value)
                println(rpad(string(name), 20), " = []")
            else
                println(rpad(string(name), 20), " = last: ", value[end])
            end

        else
            println(rpad(string(name), 20), " = ", value)
        end
    end
end

"""On each iteration, it is necessary to build MPS from the current feasible samples, which are stored in the matrix `T`. 
This function performs this step by computing link and site charges, indices, and initializing the MPS accordingly."""
function build_mps_from_feasible_samples(opt_problem::OptimizationProblem, T::Matrix{Int}, link_degeneracy::Int, diversify::Bool=false)
    A, b, x_domains, site_cards = opt_problem.A, opt_problem.b, opt_problem.x_domains, opt_problem.site_card
    link_charges = compute_link_charges(A, b, T)
    site_charges = compute_site_charges(A, x_domains)
    link_indices, site_indices = compute_indices(link_charges, site_charges, b, x_domains, A)
    mps = init_u1_mps(Float64, link_indices, site_indices, A, b, link_degeneracy, site_cards)
    if diversify
        diversify!(mps)
    end
    return mps
end

"""Completes one training step (several sweeps) for mps"""
function train_step!(mps::U1MPS, T::Matrix{Int}, solver_params::SolverParams, temperature::Float64, cost_function::Function, c_min::Float64)
    t_params = TrainParams(solver_params.LEARNING_RATE, 10^4)  ##TODO: refactor constants;
    z = sum([exp(-(cost_function(x) - c_min) / temperature) for x in eachcol(T)])
    probs = [boltzman_probability(cost_function, x_vec, temperature, z, c_min) for x_vec in eachcol(T)]
    train_nondeg!(mps, solver_params.NUM_SWEEP_ITER, T, probs, t_params)
    normalize!(mps)
end

function pick_diverse_data!(mps::U1MPS, new_feasible::Matrix{Int},
    global_charges_memorized::Vector{Set{Vector{Int}}}, graph_dist_global!::Function, sorting_crit::String,
    cost_function::Function,
    sp::SolverParams, barrier::Bool, k::Int; mixed_proportion::Float64=0.5, rank_weight::Float64=0.5)

    num_principal_new = -1  # Default value in case graph distance is not calculated;
    max_graph_dist = -1  # Default value in case graph distance is not calculated;
    min_graph_dist = -1  # Default value in case graph distance is not calculated;

    num_to_keep_worst = floor(Int, sp.KEEP_NUM_WORST * sp.NUM_FEASIBLE_SAMPLES)
    num_new = sp.NUM_FEASIBLE_SAMPLES
    @assert num_new > num_to_keep_worst
    num_best_new = num_new - num_to_keep_worst

    # Memory for graph calculation:
    n_threads = Threads.nthreads()
    BUFFERS = [Vector{Int}(undef, length(first(mps.LinkIndices[1].Charges))) for _ in 1:n_threads]
    buffer_pool = Channel{Int}(n_threads)
    for i in 1:n_threads
        put!(buffer_pool, i)
    end

    if barrier
        graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
        id = take!(buffer_pool)
        buffer = BUFFERS[id]
        try
            graph_dist_global!(buffer, col, mps, global_charges_memorized)
        finally
            put!(buffer_pool, id)
        end
    end
        num_principal_new = count(!iszero, graph_dists)
        if num_principal_new > num_new
            mask = findall(!iszero, graph_dists)
            max_graph_dist = maximum(graph_dists)
            graph_dists = graph_dists[mask]
            min_graph_dist = minimum(graph_dists)
            new_feasible = view(new_feasible, :, mask)
        else
            min_graph_dist, max_graph_dist = extrema(graph_dists)
        end

    end

    if sorting_crit == "graph_dist"
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
        id = take!(buffer_pool)
        buffer = BUFFERS[id]
        try
            graph_dist_global!(buffer, col, mps, global_charges_memorized)
        finally
            put!(buffer_pool, id)
        end
        end
    end
        num_principal_new = count(!iszero, graph_dists)
        new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, graph_dists, rev=true)
        min_graph_dist, max_graph_dist = extrema(graph_dists)

    elseif sorting_crit == "best_cost"
        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(new_feasible))
        new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, cost_values, rev=false)

    elseif sorting_crit == "random"
        cost_values = ThreadsX.map(col -> 1, eachcol(new_feasible))
        new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, cost_values, rev=true)

    elseif sorting_crit == "alternating"
        if iseven(k)
            if !barrier
                graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
                id = take!(buffer_pool)
                buffer = BUFFERS[id]
                try
                    graph_dist_global!(buffer, col, mps, global_charges_memorized)
                finally
                    put!(buffer_pool, id)
                end
            end
            end
            num_principal_new = count(!iszero, graph_dists)
            new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, graph_dists, rev=true)
            min_graph_dist, max_graph_dist = extrema(graph_dists)
        else
            sorting_function = x -> cost_function(x)
            cost_values = ThreadsX.map(col -> sorting_function(col), eachcol(new_feasible))
            new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, cost_values, rev=false)
        end
        
    elseif sorting_crit == "linear annealing"
        alpha = max(0.0, 1.0 - k/sp.NUM_GLOBAL_ITER)
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
            id = take!(buffer_pool)
            buffer = BUFFERS[id]
            try
                graph_dist_global!(buffer, col, mps, global_charges_memorized)
            finally
                put!(buffer_pool, id)
            end
        end
        end
        num_principal_new = count(!iszero, graph_dists)
        min_graph_dist, max_graph_dist = extrema(graph_dists)
        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(new_feasible))
        mixed_values = graph_dists .* (-alpha) .+ (1 - alpha) .* cost_values
        new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, mixed_values, rev=false)

    elseif sorting_crit == "mixed"
        num_graph = round(Int, num_new * mixed_proportion)
        num_cost = num_new - num_graph
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
            id = take!(buffer_pool)
            buffer = BUFFERS[id]
            try
                graph_dist_global!(buffer, col, mps, global_charges_memorized)
            finally
                put!(buffer_pool, id)
            end
    end
        end
        perm = sortperm(graph_dists, rev=true)
        nf_sorted = view(new_feasible, :, perm)
        new_feasible_graph = nf_sorted[:, 1:num_graph]

        nf_remaining = view(nf_sorted, :, num_graph+1:size(nf_sorted, 2))
        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(nf_remaining))

        new_feasible_cost = pick_best_samples(nf_remaining, num_cost, cost_values, rev=false)
        new_feasible = hcat(new_feasible_cost, new_feasible_graph)
        max_graph_dist = graph_dists[perm[1]]
        min_graph_dist = graph_dists[perm[end]]

        # Calculate principaly new samples:
        num_principal_new = searchsortedfirst(view(graph_dists, perm), 0; rev=true) - 1

    elseif sorting_crit == "mixed annealing"
        alpha = max(0.0, 0.6 * (1 - k/sp.NUM_GLOBAL_ITER))
        num_graph = round(Int, alpha * num_new)
        num_cost = num_new - num_graph
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
            id = take!(buffer_pool)
            buffer = BUFFERS[id]
            try
                graph_dist_global!(buffer, col, mps, global_charges_memorized)
            finally
                put!(buffer_pool, id)
            end
        end
        end
        perm = sortperm(graph_dists, rev=true)
        nf_sorted = view(new_feasible, :, perm)
        new_feasible_graph = nf_sorted[:, 1:num_graph]
        nf_remaining = view(nf_sorted, :, num_graph+1:size(nf_sorted, 2))
        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(nf_remaining))
        new_feasible_cost = pick_best_samples(nf_remaining, num_cost, cost_values, rev=false)
        new_feasible = hcat(new_feasible_cost, new_feasible_graph)
        max_graph_dist = graph_dists[perm[1]]
        min_graph_dist = graph_dists[perm[end]]

        # Calculate principaly new samples:
        num_principal_new = searchsortedfirst(view(graph_dists, perm), 0; rev=true) - 1

    elseif sorting_crit == "best nonzero mixed"
        num_graph = round(Int, num_new * mixed_proportion)
        num_cost  = num_new - num_graph
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
                id = take!(buffer_pool)
                buffer = BUFFERS[id]
                try
                    graph_dist_global!(buffer, col, mps, global_charges_memorized)
                finally
                    put!(buffer_pool, id)
                end
            end
        end

        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(new_feasible))
        perm_cost = sortperm(cost_values; rev=false)
        best_cost = view(new_feasible, :, perm_cost[1:num_cost])

        remaining_idx = perm_cost[num_cost+1:end]
        positive_idx = filter(i -> graph_dists[i] > 0, remaining_idx)

        if isempty(positive_idx)
            # fallback: 
            new_feasible = new_feasible[:, perm_cost[1:num_new]]
            max_graph_dist = 0
            min_graph_dist = 0
            num_principal_new = 0
        else
            cost_pos = cost_values[positive_idx]
            perm_pos = sortperm(cost_pos; rev=false)
            num_graph = min(num_graph, length(perm_pos))
            best_graph = view(new_feasible, :, positive_idx[perm_pos[1:num_graph]])
            new_feasible = hcat(best_cost, best_graph)
            max_graph_dist, min_graph_dist = extrema(graph_dists)
        end

    elseif sorting_crit == "weighted rank"
        if !barrier
            graph_dists = ThreadsX.map(eachcol(new_feasible)) do col
            id = take!(buffer_pool)
            buffer = BUFFERS[id]
            try
                graph_dist_global!(buffer, col, mps, global_charges_memorized)
            finally
                put!(buffer_pool, id)
            end
        end
        end
        num_principal_new = count(!iszero, graph_dists)
        min_graph_dist, max_graph_dist = extrema(graph_dists)
        cost_values = ThreadsX.map(col -> cost_function(col), eachcol(new_feasible))

        graph_ranks = invperm(sortperm(graph_dists, rev=true))
        cost_ranks = invperm(sortperm(cost_values, rev=false))

        mixed_values = cost_ranks .+ (rank_weight .* graph_ranks)
        new_feasible = pick_mixed_samples(new_feasible, num_best_new, num_to_keep_worst, mixed_values, rev=false)
        
    else
        throw(ArgumentError("Incorrect sorting_crit: ", sorting_crit))
    end

    return new_feasible, num_principal_new, max_graph_dist, min_graph_dist, num_to_keep_worst
end


"""
Sorts samples in T according to the cost function and keeps only num_to_keep best samples. Returns sorted and truncated T and corresponding costs.
"""
function sort_and_truncate(T::Matrix{Int}, cost_function::Function, num_to_keep::Int)
    n_cols = size(T, 2)
    costs = [cost_function(view(T, :, i)) for i in 1:n_cols]
    perm = partialsortperm(costs, 1:num_to_keep)
    return T[:, perm], costs[perm]
end 

"""
Adds worst samples from T to the best ones for diversification. The number of worst samples to keep is determined by the KEEP_NUM_WORST parameter in SolverParams. Returns updated T and the number of worst samples kept.
"""
function add_worst_samples(T::Matrix{Int}, sp::SolverParams)
    num_to_keep_worst = floor(Int, sp.KEEP_NUM_WORST * sp.NUM_FEASIBLE_SAMPLES)
    if num_to_keep_worst > 0
        worst_T = T[:, end - num_to_keep_worst + 1:end]
        T = hcat(T[:, 1:min(sp.NUM_FEASIBLE_SAMPLES, size(T, 2))], worst_T)

    else
        T = T[:, 1:min(sp.NUM_FEASIBLE_SAMPLES, size(T, 2))]
    end
    return T, num_to_keep_worst
end



function solve(opt_problem::OptimizationProblem, feasible_sampler!::Function, solver_params::SolverParams, sorting_crit::String; 
    diversify=false, barrier=false, mixed_proportion::Float64=0.5, rank_weight::Float64=0.5, parallel::Bool=true, print_stats::Bool=true, debug::Bool=false)

    # Global charges updater:
    function UpdateGlobalCharges!(mps::U1MPS{T, N}) where {T<:Integer, N<:AbstractFloat}
        for i in 1:length(mps.LinkIndices) - 1
            union!(global_charges_memorized[i], mps.LinkIndices[i + 1].Charges)
        end
    end

    sp = solver_params
    cost_function = opt_problem.cost_function
    n_vars, _ = opt_problem.num_vars, opt_problem.num_constraints

    # Global charges storage:
    global_charges_memorized = [Set{Vector{Int}}() for _ in 1:n_vars]
    # Metrics
    solver_stats = SolverStatistics()

    # Initial feasible dataset
    T = zeros(Int, n_vars, sp.NUM_FEASIBLE_SAMPLES)
    feasible_sampler!(T)
    T, costs = sort_and_truncate(T, cost_function, sp.NUM_FEASIBLE_SAMPLES)

    num_to_keep = floor(Int, sp.UTILITY_FRACTION * size(T, 2))
    best_samples = T[:, 1:num_to_keep]
    best_costs = mapslices(cost_function, best_samples, dims=1)
    solver_stats.c_min = minimum(best_costs)
    solver_stats.incub = best_samples[:, 1]

    push!(solver_stats.learning_curve, solver_stats.c_min)
    push!(solver_stats.utility_curve, mean(best_costs))

    # Memory BUF for samples from MPS to process:
    samples = Matrix{Int}(undef, n_vars, sp.NUM_MPS_SAMPLES + size(best_samples, 2))  #TODO: memory allocation;
    # Memory Buf for samples from feasible sampler:
    MAX_SAMPLE_SIZE = 20000  # num samples from another sample source (not MPS)
    @assert MAX_SAMPLE_SIZE > sp.NUM_MPS_SAMPLES # assert, that enough memory;
    X = zeros(Int, n_vars, MAX_SAMPLE_SIZE)

    if print_stats
        println("Initial c_min: ", solver_stats.c_min)
        println("Initial utility: ", solver_stats.utility_curve[end])
        println("Strategy for picking new samples: ", sorting_crit)
        println("Diversification: ", diversify)
        println("Barrier: ", barrier)
    end

    # Main loop
    for k in 1:sp.NUM_GLOBAL_ITER

        if print_stats
            println("\n=== Global Iteration ", k, " ===")
        end

        # MPS initialization (with diversification if specified)
        mps = build_mps_from_feasible_samples(opt_problem, T, sp.LINK_DEGENERACY, diversify)
        orthogonalize!(mps)
        solver_stats.num_samples_in_mps = Int(round(u1_norm(mps) ^ 2))
        normalize!(mps)

        # Learning step
        temperature = std(costs)
        train_step!(mps, T, sp, temperature, cost_function, solver_stats.learning_curve[end])
        solver_stats.temperature = temperature

        # Sampling
        fill!(samples, 0)  # clear memory buf;
        if parallel
            sample_nondeg_parallel!(mps, samples, sp.NUM_MPS_SAMPLES)
        else
            sample_nondeg!(mps, samples, sp.NUM_MPS_SAMPLES)
        end

        # Keep best samples from previous iteration
        for (j, best_x) in enumerate(eachcol(best_samples))
            samples[:, sp.NUM_MPS_SAMPLES + j] .= best_x
        end

        # Unique samples
        T = unique(eachcol(samples)) |> x -> reduce(hcat, x)
        solver_stats.num_unique_samples = size(T, 2)

        # Sort and keep best feasible, then - add worst samples for diversification;
        T, costs = sort_and_truncate(T, cost_function, size(T, 2))
        best_samples = T[:, 1:num_to_keep]
        best_costs = mapslices(cost_function, best_samples, dims=1)
        solver_stats.c_min = minimum(best_costs)
        solver_stats.incub = best_samples[:, 1]

        # Update learning curve and statistics:
        push!(solver_stats.learning_curve, solver_stats.c_min)
        push!(solver_stats.utility_curve, mean(best_costs))

        # Add new feasible samples for diversification
        UpdateGlobalCharges!(mps)  # Write info about current charges;

        if debug
            println("Worst cost in mps: ", cost_function(T[:, end]))
            println("Worst cost in mps may be taken: ", cost_function(T[:, min(size(T, 2), sp.NUM_FEASIBLE_SAMPLES)]))
        end

        X[:, 1:size(T, 2)] .= T[:, 1:end]  # copy mps samples to X
        X[:, size(T, 2)+1:end] .= 0  # fill zeros for feasible data (must be done for assignment feasible sampler;)
        feasible_sampler!(@view X[:, size(T, 2)+1:end])  # add random feasible samples for diversification;

        if debug
            println("Best random cost: ", minimum([cost_function(x) for x in eachcol(X[:, size(T, 2)+1:end])]))
            @assert allcols_AX_equal_b(opt_problem.A, X, opt_problem.b)  # Make sure, that we are in a feasible sub-space;
        end

        T, 
        solver_stats.num_principal_new, 
        solver_stats.max_graph_dist, 
        solver_stats.min_graph_dist, 
        solver_stats.num_to_keep_worst = pick_diverse_data!(mps, X,  global_charges_memorized, graph_dist_global!, sorting_crit, cost_function, sp, barrier, k, mixed_proportion=mixed_proportion, rank_weight=rank_weight)
        
        if debug
            @assert allcols_AX_equal_b(opt_problem.A, T, opt_problem.b)
        end

        best_set = Set(eachcol(best_samples))
        keep_indices = filter(i -> view(T, :, i) ∉ best_set, axes(T, 2))
        T = hcat(best_samples, view(T, :, keep_indices))  # Not to loose best (in cost) samples;
        T, costs = sort_and_truncate(T, cost_function, size(T, 2))
        best_samples = T[:, 1:num_to_keep]

        if print_stats
            print_statistics(solver_stats)
        end

    end

    return solver_stats
end