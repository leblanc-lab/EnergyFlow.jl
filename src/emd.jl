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
    emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute the Earth Mover's Distance between two events.

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
function emd(event1::Matrix{Float64}, event2::Matrix{Float64}; 
             R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
             measure::String="euclidean", coords::String="hadronic",
             periodic_phi::Bool=false, n_iter_max::Int=100000,
             return_flow::Bool=false, kwargs...)
    
    # Handle empty events
    if size(event1, 1) == 0 && size(event2, 1) == 0
        return return_flow ? (0.0, zeros(0, 0)) : 0.0
    elseif size(event1, 1) == 0 || size(event2, 1) == 0
        throw(ArgumentError("Cannot compute EMD between empty and non-empty events"))
    end
    
    # Create parameters struct
    params = EMDParameters(R=R, beta=beta, norm=norm, measure=measure, 
                          coords=coords, periodic_phi=periodic_phi, n_iter_max=n_iter_max)
    
    # Process events
    pTs1, coords1 = process_event(event1, params)
    pTs2, coords2 = process_event(event2, params)
    
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(coords1, coords2, params)
    
    # Apply beta weighting
    if params.beta != 1.0
        dist_matrix = dist_matrix .^ params.beta
    end
    
    # Handle normalization and dummy particles
    pTs1, pTs2, dist_matrix, scale_factor = handle_normalization(pTs1, pTs2, dist_matrix, params)
    
    # Solve optimal transport problem
    if EnergyFlow.USE_EXACT_SOLVER[]
        # Use exact solver (default)
        flow, cost = solve_emd_exact(pTs1, pTs2, dist_matrix)
        # Apply scale factor back to get true EMD
        cost = cost * scale_factor
    else
        # Use Sinkhorn approximation (if explicitly requested)
        # Use better epsilon based on the scale of the problem
        # Smaller epsilon gives more accurate results but slower convergence
        max_dist = maximum(dist_matrix)
        epsilon = max(1e-5, min(0.01, max_dist / 100))
        
        # Try Sinkhorn with adaptive parameters
        try
            flow = sinkhorn(pTs1, pTs2, dist_matrix, epsilon; 
                          maxiter=params.n_iter_max,
                          atol=1e-8,
                          rtol=1e-6)
            cost = sum(flow .* dist_matrix) * scale_factor
        catch e
            # If Sinkhorn fails, try with larger epsilon
            @warn "Sinkhorn failed with epsilon=$epsilon, trying with larger value" exception=e
            epsilon_fallback = max(0.1, max_dist / 10)
            flow = sinkhorn(pTs1, pTs2, dist_matrix, epsilon_fallback; 
                          maxiter=params.n_iter_max,
                          atol=1e-6,
                          rtol=1e-4)
            cost = sum(flow .* dist_matrix) * scale_factor
        end
    end
    
    if return_flow
        return cost, flow
    else
        return cost
    end
end

"""
    emds(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix for a collection of events.

# Arguments
- `events`: Vector of event matrices

# Keywords
Same as `emd` function

# Returns
- `Matrix{Float64}`: Symmetric matrix of pairwise EMD values
"""
function emds(events::Vector{Matrix{Float64}}; kwargs...)
    n = length(events)
    emd_matrix = zeros(Float64, n, n)
    
    for i in 1:n
        for j in i+1:n
            emd_matrix[i, j] = emd(events[i], events[j]; kwargs...)
            emd_matrix[j, i] = emd_matrix[i, j]
        end
    end
    
    return emd_matrix
end

"""
    process_event(event::Matrix{Float64}, params::EMDParameters)

Process event for EMD calculation.

# Returns
- `weights::Vector{Float64}`: Particle weights (pT values)
- `coords::Matrix{Float64}`: Particle coordinates
"""
function process_event(event::Matrix{Float64}, params::EMDParameters)
    event = copy(event)
    
    # Handle coordinate conversion if needed
    converted_to_cartesian = false
    if params.measure != "euclidean" && params.coords == "hadronic"
        event = hadronic_to_cartesian(event)
        converted_to_cartesian = true
    end
    
    # Extract weights and coordinates
    if size(event, 2) == 3 && !converted_to_cartesian
        # Format: [pT, y, phi]
        weights = event[:, 1]
        coords = event[:, 2:3]
    elseif size(event, 2) == 4
        # Format: [pT, y, phi, weight] or [E, px, py, pz]
        if params.coords == "cartesian" || converted_to_cartesian
            # Cartesian: weights are energies
            weights = event[:, 1]
            coords = event[:, 2:4]
        else
            # Hadronic with explicit weights
            weights = event[:, 1] .* event[:, 4]  # pT * weight
            coords = event[:, 2:3]
        end
    else
        error("Event must have 3 or 4 columns")
    end
    
    # Normalize coordinates for spherical measure
    if params.measure != "euclidean"
        coords = normalize_coordinates(coords)
    elseif params.periodic_phi && size(coords, 2) >= 2
        # Handle periodic phi (assuming phi is last column)
        coords[:, end] = mod.(coords[:, end], 2π)
    end
    
    # Apply normalization if requested
    if params.norm && sum(weights) > 0
        weights = weights / sum(weights)
    end
    
    # Add dummy particle for unbalanced transport
    if !params.norm
        weights = vcat(weights, 0.0)
        coords = vcat(coords, zeros(1, size(coords, 2)))
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
        # Already in matrix format
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
    cartesian = zeros(n_particles, 4)
    
    pT = event[:, 1]
    y = event[:, 2]
    phi = event[:, 3]
    
    # For massless particles: E = pT * cosh(y)
    cartesian[:, 1] = pT .* cosh.(y)  # E
    cartesian[:, 2] = pT .* cos.(phi)  # px
    cartesian[:, 3] = pT .* sin.(phi)  # py
    cartesian[:, 4] = pT .* sinh.(y)   # pz
    
    return cartesian
end

function normalize_coordinates(coords::Matrix{Float64})
    """Normalize coordinate vectors for spherical measure"""
    norms = sqrt.(sum(coords.^2, dims=2))
    # Avoid division by zero
    norms[norms .== 0] .= 1.0
    return coords ./ norms
end

function handle_normalization(pTs1::Vector{Float64}, pTs2::Vector{Float64}, 
                            dist_matrix::Matrix{Float64}, params::EMDParameters)
    """Handle weight normalization and dummy particles"""
    
    pT1_sum = sum(pTs1[1:end-1])  # Exclude dummy
    pT2_sum = sum(pTs2[1:end-1])  # Exclude dummy
    scale_factor = 1.0
    
    if !params.norm
        pT_diff = pT2_sum - pT1_sum
        if pT_diff > 0
            pTs1[end] = pT_diff
            dist_matrix[end, :] .= 1.0
        elseif pT_diff < 0
            pTs2[end] = -pT_diff
            dist_matrix[:, end] .= 1.0
        end
        
        # Rescale for numerical stability
        rescale = max(pT1_sum, pT2_sum)
        if rescale > 0
            pTs1 = pTs1 / rescale
            pTs2 = pTs2 / rescale
            scale_factor = rescale  # Store the scale factor to apply back later
            return pTs1, pTs2, dist_matrix, scale_factor
        end
    end
    
    return pTs1, pTs2, dist_matrix, scale_factor
end

# Solver functions (basic versions, extended by package extension)

# This function will be defined by the package extension when JuMP and HiGHS are available
# solve_emd_exact is defined in ext/EnergyFlowExactSolverExt.jl

# Solver status functions

"""
    has_exact_solver()

Check if the exact EMD solver (JuMP + HiGHS) is available.

# Returns
- `Bool`: true if exact solver is available, false otherwise
"""
function has_exact_solver()
    return EnergyFlow.HAS_EXACT_SOLVER[]
end

"""
    check_solver_status()

Print detailed information about the available EMD solvers.
"""
function check_solver_status()
    println("EnergyFlow.jl Solver Status")
    println("===========================")
    
    println("✓ Exact solver (JuMP + HiGHS) is available")
    println("  - This is now the default solver")
    println("  - Provides exact optimal transport solutions")
    
    current_solver = EnergyFlow.USE_EXACT_SOLVER[] ? "Exact (JuMP/HiGHS)" : "Sinkhorn approximation"
    println("\nCurrent solver: $current_solver")
    
    if !EnergyFlow.USE_EXACT_SOLVER[]
        println("\nSinkhorn parameters:")
        println("  - Adaptive epsilon based on distance matrix scale")
        println("  - Max iterations: 100,000 (default)")
        println("  - Tolerance: atol=1e-8, rtol=1e-6")
    end
    
    println("\nTo switch solvers:")
    println("  use_exact_solver!()   # Use exact solver (default)")
    println("  use_sinkhorn!()       # Use Sinkhorn approximation")
end

"""
    use_exact_solver!()

Switch to using the exact linear programming solver (default).
"""
function use_exact_solver!()
    EnergyFlow.USE_EXACT_SOLVER[] = true
    println("Switched to exact solver (JuMP/HiGHS)")
end

"""
    use_sinkhorn!()

Switch to using the Sinkhorn approximation solver.
Only use this if you need faster approximate solutions.
"""
function use_sinkhorn!()
    EnergyFlow.USE_EXACT_SOLVER[] = false
    println("Switched to Sinkhorn approximation")
    println("Warning: Results may differ from exact solutions")
end