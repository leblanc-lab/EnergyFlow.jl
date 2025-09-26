# EMD (Earth Mover's Distance) implementation

"""
    EMDParameters

Parameters for EMD computation.

# Fields
- `R::Float64`: R parameter controlling relative importance of terms (default: 1.0)
- `beta::Float64`: Angular weighting exponent for distance matrix (default: 1.0)
- `norm::Bool`: Whether to normalize particle weights to sum to 1 (default: false)
- `measure::String`: Distance metric ("euclidean" or "spherical") (default: "euclidean")
- `coords::String`: Coordinate system ("hadronic" or "cartesian") (default: "hadronic")
- `periodic_phi::Bool`: Handle phi coordinate periodicity (default: false)
- `n_iter_max::Int`: Maximum iterations for solver (default: 100000)
"""
Base.@kwdef struct EMDParameters
    R::Float64 = 1.0
    beta::Float64 = 1.0
    norm::Bool = false
    measure::String = "euclidean"
    coords::String = "hadronic"
    periodic_phi::Bool = false
    n_iter_max::Int = 100000
end


"""
    emd_exact(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute the Earth Mover's Distance between two events using the exact HiGHS solver.

# Arguments
- `event1`: First event as (M, 3) or (M, 4) matrix where M is multiplicity
- `event2`: Second event as (N, 3) or (N, 4) matrix where N is multiplicity

For 3-column format: [pT, y, phi]
For 4-column format: [pT, y, phi, weight]

# Keywords
- `R::Float64=1.0`: R parameter controlling relative importance of terms
- `beta::Float64=1.0`: Angular weighting exponent for distance matrix
- `norm::Bool=false`: Whether to normalize particle weights to sum to 1
- `measure::String="euclidean"`: Distance metric ("euclidean" or "spherical")
- `coords::String="hadronic"`: Coordinate system ("hadronic" or "cartesian")
- `periodic_phi::Bool=false`: Handle phi coordinate periodicity
- `n_iter_max::Int=100000`: Maximum iterations for solver
- `return_flow::Bool=false`: Whether to return optimal transport flow matrix

# Returns
- `Float64`: EMD value
- `Matrix{Float64}` (optional): Flow matrix if return_flow=true
"""
function emd_exact(event1::Matrix{Float64}, event2::Matrix{Float64};
                   R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                   measure::String="euclidean", coords::String="hadronic",
                   periodic_phi::Bool=false, n_iter_max::Int=100000,
                   return_flow::Bool=false, kwargs...)

    # Validate that R and beta are positive
    if R <= 0
        throw(ArgumentError("R must be a positive value"))
    end
    if beta <= 0
        throw(ArgumentError("beta must be a positive value"))
    end

    # Empty events
    if size(event1, 1) == 0 && size(event2, 1) == 0
        return return_flow ? (0.0, zeros(0, 0)) : 0.0
    elseif size(event1, 1) == 0 || size(event2, 1) == 0
        throw(ArgumentError("Cannot compute EMD between empty and non-empty events"))
    end
    
    params = EMDParameters(R=R, beta=beta, norm=norm, measure=measure, 
                          coords=coords, periodic_phi=periodic_phi, n_iter_max=n_iter_max)
    
    pTs1, coords1 = process_event(event1, params)
    pTs2, coords2 = process_event(event2, params)
    
    dist_matrix = compute_distance_matrix(coords1, coords2, params)
    
    if params.beta != 1.0
        dist_matrix = dist_matrix .^ params.beta
    end
    
    pTs1, pTs2, dist_matrix, scale_factor = handle_normalization(pTs1, pTs2, dist_matrix, params)

    # Always use HiGHS exact solver
    flow, cost = solve_emd_exact(pTs1, pTs2, dist_matrix)
    cost = cost * scale_factor
    
    if return_flow
        return cost, flow
    else
        return cost
    end
end

"""
    emds_exact(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix for a collection of events using the exact HiGHS solver.

# Arguments
- `events`: Vector of event matrices, each as (M, 3) or (M, 4) matrix

# Keywords
Same as `emd_exact` function

# Returns
- `Matrix{Float64}`: Symmetric matrix of pairwise EMDs
"""
function emds_exact(events::Vector{Matrix{Float64}}; kwargs...)
    n = length(events)
    emd_matrix = zeros(Float64, n, n)

    for i in 1:n
        for j in i+1:n
            emd_matrix[i, j] = emd_exact(events[i], events[j]; kwargs...)
            emd_matrix[j, i] = emd_matrix[i, j]
        end
    end

    return emd_matrix
end

"""
    process_event(event::Matrix{Float64}, params::EMDParameters)

Process event for EMD calculation using views to minimize allocations.

# Returns
- `weights::Vector{Float64}`: Particle weights (pT values)
- `coords::Matrix{Float64}`: Particle coordinates
"""
function process_event(event::Matrix{Float64}, params::EMDParameters)
    n_particles = size(event, 1)
    n_cols = size(event, 2)
    
    # Check if conversion to Cartesian coordinates is needed
    needs_cartesian = params.measure != "euclidean" && params.coords == "hadronic"
    
    if needs_cartesian
        # Must convert hadronic to cartesian - requires allocation
        cartesian = hadronic_to_cartesian(event)
        return process_cartesian_event(cartesian, params, n_particles)
    end
    
    # Determine coordinate dimensions
    if n_cols == 3
        # [pT, y, phi] format
        if params.norm
            # Need to normalize - must allocate
            weights = event[:, 1] / sum(event[:, 1])
            coords = event[:, 2:3]
        else
            # Add dummy particle - must allocate for extended arrays
            weights = vcat(event[:, 1], 0.0)
            coords = vcat(event[:, 2:3], zeros(1, 2))
        end
    elseif n_cols == 4
        if params.coords == "cartesian"
            # [E, px, py, pz] format
            if params.norm
                weights = event[:, 1] / sum(event[:, 1])
                coords = event[:, 2:4]
            else
                weights = vcat(event[:, 1], 0.0)
                coords = vcat(event[:, 2:4], zeros(1, 3))
            end
        else
            # [pT, y, phi, weight] format
            effective_weights = event[:, 1] .* event[:, 4]
            if params.norm && sum(effective_weights) > 0
                weights = effective_weights / sum(effective_weights)
                coords = event[:, 2:3]
            else
                weights = vcat(effective_weights, 0.0)
                coords = vcat(event[:, 2:3], zeros(1, 2))
            end
        end
    else
        error("Event must have 3 or 4 columns")
    end
    
    # Handle coordinate transformations
    if params.measure != "euclidean"
        coords = normalize_coordinates(coords)
    elseif params.periodic_phi && size(coords, 2) >= 2
        # In-place modification for periodic phi
        coords = copy(coords)  # Need to copy to avoid modifying original
        @inbounds @simd for i in 1:size(coords, 1)
            coords[i, end] = mod(coords[i, end], 2π)
        end
    end
    
    return weights, coords
end

"""
    process_cartesian_event(cartesian::Matrix{Float64}, params::EMDParameters, n_particles::Int)

Process cartesian event data.
"""
function process_cartesian_event(cartesian::Matrix{Float64}, params::EMDParameters, n_particles::Int)
    # Extract energy as weights and spatial coordinates
    if params.norm
        weights = cartesian[:, 1] / sum(cartesian[:, 1])
        coords = cartesian[:, 2:4]
    else
        weights = vcat(cartesian[:, 1], 0.0)
        coords = vcat(cartesian[:, 2:4], zeros(1, 3))
    end
    
    if params.measure != "euclidean"
        coords = normalize_coordinates(coords)
    end
    
    return weights, coords
end

"""
    prepare_event_for_emd(event; pT_col=1, y_col=2, phi_col=3, weight_col=nothing)

Prepare event data in the format expected by EMD functions.

# Arguments
- `event`: Event data (can be various formats)
- `pT_col`: Column index for pT (default: 1)
- `y_col`: Column index for rapidity (default: 2)
- `phi_col`: Column index for azimuthal angle (default: 3)
- `weight_col`: Column index for weights (default: nothing)

# Returns
- `Matrix{Float64}`: Event in standard format [pT, y, phi] or [pT, y, phi, weight]
"""
function prepare_event_for_emd(event; pT_col=1, y_col=2, phi_col=3, weight_col=nothing)
    if isa(event, Matrix)
        if weight_col === nothing
            return event[:, [pT_col, y_col, phi_col]]
        else
            return event[:, [pT_col, y_col, phi_col, weight_col]]
        end
    else
        error("Unsupported event format")
    end
end

# Helper functions

function hadronic_to_cartesian(event::Matrix{Float64})
    """Convert hadronic coordinates (pT, y, phi) to cartesian (E, px, py, pz)"""
    n_particles = size(event, 1)
    cartesian = Matrix{Float64}(undef, n_particles, 4)
    
    pT = @view event[:, 1]
    y = @view event[:, 2]
    phi = @view event[:, 3]
    
    # Pre-compute expensive trig functions
    cosh_y = cosh.(y)
    sinh_y = sinh.(y)
    cos_phi = cos.(phi)
    sin_phi = sin.(phi)
    
    @inbounds @simd for i in 1:n_particles
        cartesian[i, 1] = pT[i] * cosh_y[i]   # E
        cartesian[i, 2] = pT[i] * cos_phi[i]  # px
        cartesian[i, 3] = pT[i] * sin_phi[i]  # py
        cartesian[i, 4] = pT[i] * sinh_y[i]   # pz
    end
    
    return cartesian
end

function normalize_coordinates(coords::Matrix{Float64})
    """Normalize coordinate vectors for spherical measure"""
    n_particles = size(coords, 1)
    n_dims = size(coords, 2)
    normalized = similar(coords)
    
    @inbounds for i in 1:n_particles
        norm_sq = 0.0
        @simd for j in 1:n_dims
            norm_sq += coords[i, j]^2
        end
        
        norm = sqrt(norm_sq)
        if norm > 0
            inv_norm = 1.0 / norm
            @simd for j in 1:n_dims
                normalized[i, j] = coords[i, j] * inv_norm
            end
        else
            @simd for j in 1:n_dims
                normalized[i, j] = coords[i, j]
            end
        end
    end
    
    return normalized
end

function handle_normalization(pTs1::AbstractVector{Float64}, pTs2::AbstractVector{Float64}, 
                            dist_matrix::AbstractMatrix{Float64}, params::EMDParameters)
    """Handle weight normalization and dummy particles - optimized in-place version"""
    
    n1 = params.norm ? length(pTs1) : length(pTs1) - 1
    n2 = params.norm ? length(pTs2) : length(pTs2) - 1
    
    pT1_sum = sum(@view pTs1[1:n1])
    pT2_sum = sum(@view pTs2[1:n2])
    scale_factor = 1.0
    
    if !params.norm
        pT_diff = pT2_sum - pT1_sum
        if pT_diff > 0
            pTs1[end] = pT_diff
            @inbounds @simd for j in 1:size(dist_matrix, 2)
                dist_matrix[end, j] = 1.0
            end
        elseif pT_diff < 0
            pTs2[end] = -pT_diff
            @inbounds @simd for i in 1:size(dist_matrix, 1)
                dist_matrix[i, end] = 1.0
            end
        end
        
        rescale = max(pT1_sum, pT2_sum)
        if rescale > 0
            # In-place normalization
            inv_rescale = 1.0 / rescale
            @inbounds @simd for i in eachindex(pTs1)
                pTs1[i] *= inv_rescale
            end
            @inbounds @simd for i in eachindex(pTs2)
                pTs2[i] *= inv_rescale
            end
            scale_factor = rescale
        end
    end
    
    return pTs1, pTs2, dist_matrix, scale_factor
end

# Solver functions

# solve_emd_exact is defined in src/exact_solver.jl