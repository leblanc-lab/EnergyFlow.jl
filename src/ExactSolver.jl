# ExactSolver.jl - Exact EMD solver using JuMP and HiGHS

module ExactSolver

using JuMP
using HiGHS
using LinearAlgebra
using ..Utils

# Internal distance matrix computation
function compute_distance_matrix_internal(coords1::Matrix{Float64}, coords2::Matrix{Float64}, params::EMDParameters)
    n1, n2 = size(coords1, 1), size(coords2, 1)
    C = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            if params.measure == "euclidean"
                dist = 0.0
                for k in 1:size(coords1, 2)
                    if params.periodic_phi && k == 2  # phi column
                        d_phi = abs(coords1[i, k] - coords2[j, k])
                        d_phi = min(d_phi, 2π - d_phi)
                        dist += d_phi^2
                    else
                        dist += (coords1[i, k] - coords2[j, k])^2
                    end
                end
                C[i, j] = sqrt(dist) / params.R
            else  # spherical
                dot_product = sum(coords1[i, :] .* coords2[j, :])
                C[i, j] = acos(clamp(dot_product, -1.0, 1.0)) / params.R
            end
        end
    end

    return C
end

"""
    solve_emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Solve EMD using exact linear programming with JuMP and HiGHS.

# Keywords
- `R::Float64=1.0`: Maximum distance parameter
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `return_flow::Bool=false`: Whether to return the flow matrix
- Other parameters from EMDParameters

# Returns
- `(emd_value::Float64, status::Symbol, flow_matrix::Union{Matrix{Float64},Nothing})`
"""
function solve_emd(event1::Matrix{Float64}, event2::Matrix{Float64};
                   R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                   return_flow::Bool=false, measure::String="euclidean",
                   periodic_phi::Bool=false, gdim::Int=2, mask::Bool=false,
                   kwargs...)

    # Create parameters
    params = EMDParameters(
        R=R, beta=beta, norm=norm, measure=measure,
        periodic_phi=periodic_phi, gdim=gdim, mask=mask
    )

    # Prepare events
    ev1 = prepare_event_for_emd(event1; params=params)
    ev2 = prepare_event_for_emd(event2; params=params)

    # Extract weights
    a = ev1[:, 1]
    b = ev2[:, 1]

    # Normalize weights to sum to 1 if needed
    sum_a = sum(a)
    sum_b = sum(b)

    if abs(sum_a - sum_b) > 1e-10
        # Handle unbalanced case by adding dummy particles
        if sum_a > sum_b
            push!(b, sum_a - sum_b)
            # Add dummy particle coordinates (far away)
            ev2 = vcat(ev2, [sum_a - sum_b 1000.0 1000.0])
        else
            push!(a, sum_b - sum_a)
            # Add dummy particle coordinates (far away)
            ev1 = vcat(ev1, [sum_b - sum_a 1000.0 1000.0])
        end
    end

    # Compute distance matrix
    coords1 = ev1[:, 2:end]
    coords2 = ev2[:, 2:end]
    C = compute_distance_matrix_internal(coords1, coords2, params)

    # Apply beta scaling
    if beta != 1.0
        if beta == 2.0
            C = C .^ 2
        else
            C = C .^ beta
        end
    end

    # Solve using exact LP
    G, cost, status = solve_emd_exact_lp(a, b, C)

    # Return in standard format
    flow_matrix = return_flow ? G : nothing
    return (cost, status, flow_matrix)
end

"""
    solve_emd_exact_lp(a::Vector{Float64}, b::Vector{Float64}, C::Matrix{Float64})

Core exact LP solver for EMD.
"""
function solve_emd_exact_lp(a::Vector{Float64}, b::Vector{Float64}, C::Matrix{Float64})
    n, m = length(a), length(b)

    # Create JuMP model with HiGHS optimizer
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Decision variables: flow matrix G[i,j] >= 0
    @variable(model, G[1:n, 1:m] >= 0)

    # Objective: minimize sum of costs
    @objective(model, Min, sum(G[i,j] * C[i,j] for i in 1:n, j in 1:m))

    # Constraints: supply constraints (sum over j)
    for i in 1:n
        @constraint(model, sum(G[i,j] for j in 1:m) == a[i])
    end

    # Constraints: demand constraints (sum over i)
    for j in 1:m
        @constraint(model, sum(G[i,j] for i in 1:n) == b[j])
    end

    # Solve the linear program
    optimize!(model)

    # Check solver status
    status = termination_status(model)
    if status != OPTIMAL
        if status == INFEASIBLE
            return zeros(n, m), Inf, :infeasible
        elseif status == UNBOUNDED
            return zeros(n, m), -Inf, :unbounded
        else
            return zeros(n, m), NaN, :error
        end
    end

    # Extract solution
    G_opt = value.(G)
    cost = objective_value(model)

    return G_opt, cost, :success
end

"""
    solve_emds(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix using exact solver.
"""
function solve_emds(events::Vector{Matrix{Float64}}; kwargs...)
    n = length(events)
    dist_matrix = zeros(n, n)

    for i in 1:n
        for j in (i+1):n
            emd_val, _, _ = solve_emd(events[i], events[j]; kwargs...)
            dist_matrix[i, j] = emd_val
            dist_matrix[j, i] = emd_val
        end
    end

    return dist_matrix
end

"""
    solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD between two sets of events.
"""
function solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)
    n1 = length(events1)
    n2 = length(events2)
    dist_matrix = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            emd_val, _, _ = solve_emd(events1[i], events2[j]; kwargs...)
            dist_matrix[i, j] = emd_val
        end
    end

    return dist_matrix
end

# Always available since JuMP and HiGHS are required dependencies
is_available() = true

export solve_emd, solve_emds, is_available

end # module ExactSolver