# CxxSolver.jl - C++ Wasserstein solver plugin

module CxxSolver

using ..Utils
import CxxWrap

# Module-level constants
const CXX_LIB_PATH = joinpath(@__DIR__, "libwasserstein_wrapper")
const CXX_LIB_FILE = joinpath(@__DIR__, "libwasserstein_wrapper.dylib")

# Check if library exists
const LIB_EXISTS = isfile(CXX_LIB_FILE)

# Global variables for C++ functions
global CXX_AVAILABLE = false
global compute_emd_cpp_func = nothing
global compute_emd_yphi_func = nothing
global compute_emds_cpp_func = nothing

# Try to load the library at module initialization
function __init__()
    global CXX_AVAILABLE, compute_emd_cpp_func, compute_emd_yphi_func, compute_emds_cpp_func

    if !LIB_EXISTS
        @warn "C++ library not found at $CXX_LIB_FILE. Build with: cd src/cxxwrap && ./build_wrapper.sh"
        return
    end

    try
        # Load the wrapper module into a temporary module
        @eval module CxxFuncs
            using CxxWrap
            @wrapmodule(() -> $CXX_LIB_PATH)
            @initcxx
        end

        # Get references to the functions
        compute_emd_cpp_func = CxxFuncs.compute_emd_cpp
        compute_emd_yphi_func = CxxFuncs.compute_emd_yphi
        compute_emds_cpp_func = CxxFuncs.compute_emds_cpp

        CXX_AVAILABLE = true
        @info "C++ Wasserstein wrapper loaded successfully"
    catch e
        CXX_AVAILABLE = false
        @warn "Failed to load C++ wrapper: $e"
    end
end

"""
    solve_emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Solve EMD using the C++ Wasserstein library.

# Keywords
- `R::Float64=1.0`: Maximum distance parameter
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- Other parameters are ignored for C++ solver

# Returns
- `(emd_value::Float64, status::Symbol, nothing)`
"""
function solve_emd(event1::Matrix{Float64}, event2::Matrix{Float64};
                   R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                   kwargs...)

    if !CXX_AVAILABLE || isnothing(compute_emd_yphi_func)
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end

    # Handle empty events
    if size(event1, 1) == 0 || size(event2, 1) == 0
        return (0.0, :Optimal, nothing)
    end

    # Extract weights and coordinates
    n1 = size(event1, 1)
    n2 = size(event2, 1)

    # Create flat arrays for C++
    weights1_vec = Vector{Float64}(event1[:, 1])
    coords1_vec = zeros(Float64, n1 * 2)
    for i in 1:n1
        coords1_vec[(i-1)*2 + 1] = event1[i, 2]  # y
        coords1_vec[(i-1)*2 + 2] = event1[i, 3]  # phi
    end

    weights2_vec = Vector{Float64}(event2[:, 1])
    coords2_vec = zeros(Float64, n2 * 2)
    for i in 1:n2
        coords2_vec[(i-1)*2 + 1] = event2[i, 2]  # y
        coords2_vec[(i-1)*2 + 2] = event2[i, 3]  # phi
    end

    # Call C++ function - need to convert to StdVector
    try
        # Convert Julia vectors to StdVector for C++
        weights1_std = CxxWrap.StdVector(weights1_vec)
        coords1_std = CxxWrap.StdVector(coords1_vec)
        weights2_std = CxxWrap.StdVector(weights2_vec)
        coords2_std = CxxWrap.StdVector(coords2_vec)

        # Call the C++ function
        emd_value = compute_emd_yphi_func(weights1_std, coords1_std,
                                          weights2_std, coords2_std,
                                          R, beta, norm)
        return (emd_value, :Optimal, nothing)
    catch e
        @warn "C++ EMD computation failed: $e"
        return (NaN, :Failed, nothing)
    end
end

"""
    solve_emds(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix using the C++ Wasserstein library.
This uses the batch C++ function for better performance.

# Returns
- `Matrix{Float64}`: Symmetric EMD distance matrix
"""
function solve_emds(events::Vector{Matrix{Float64}};
                    R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                    symmetric::Bool=true, kwargs...)

    if !CXX_AVAILABLE || isnothing(compute_emds_cpp_func)
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end

    n = length(events)

    if symmetric
        # Convert events to flat format expected by C++
        # Each event becomes [pT1, y1, phi1, pT2, y2, phi2, ...]
        events_flat = Vector{CxxWrap.StdVector{Float64}}()
        for event in events
            n_particles = size(event, 1)
            flat_event = zeros(Float64, n_particles * 3)
            for i in 1:n_particles
                flat_event[(i-1)*3 + 1] = event[i, 1]  # pT
                flat_event[(i-1)*3 + 2] = event[i, 2]  # y
                flat_event[(i-1)*3 + 3] = event[i, 3]  # phi
            end
            push!(events_flat, CxxWrap.StdVector(flat_event))
        end

        # Convert vector of StdVector to StdVector of StdVector
        events_std = CxxWrap.StdVector(events_flat)

        # Call batch C++ function
        try
            result_std = compute_emds_cpp_func(events_std, R, beta, norm)
            # Convert StdVector result to Julia matrix
            result = zeros(n, n)
            for i in 1:n
                for j in 1:n
                    result[i, j] = result_std[(i-1)*n + j]
                end
            end
            return result
        catch e
            @debug "Batch C++ computation failed, falling back to pairwise: $e"
            # Fallback to pairwise computation
            result = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    emd_val, _, _ = solve_emd(events[i], events[j]; R=R, beta=beta, norm=norm)
                    result[i, j] = emd_val
                    result[j, i] = emd_val
                end
            end
            return result
        end
    else
        # Non-symmetric: compute all pairs
        result = zeros(n, n)
        for i in 1:n
            for j in 1:n
                if i != j
                    emd_val, _, _ = solve_emd(events[i], events[j]; R=R, beta=beta, norm=norm)
                    result[i, j] = emd_val
                end
            end
        end
        return result
    end
end

"""
    solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD between two sets of events using C++ solver.
"""
function solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}};
                    R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                    kwargs...)

    if !CXX_AVAILABLE
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end

    n1 = length(events1)
    n2 = length(events2)
    result = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            emd_val, _, _ = solve_emd(events1[i], events2[j]; R=R, beta=beta, norm=norm)
            result[i, j] = emd_val
        end
    end

    return result
end

# Check if the solver is available
is_available() = CXX_AVAILABLE

export solve_emd, solve_emds, is_available

end # module CxxSolver