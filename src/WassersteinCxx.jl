# CxxWrap interface for Wasserstein C++ library
module WassersteinCxx

using CxxWrap

# Path to the compiled wrapper library
const WRAPPER_PATH = joinpath(@__DIR__, "libwasserstein_wrapper")

# Check if library exists, if not, provide build instructions
if !isfile(WRAPPER_PATH * (Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so"))
    @info """
    Wasserstein C++ wrapper library not found. To build it:

    1. Install required dependencies:
       - cmake (>= 3.14)
       - C++ compiler with C++14 support

    2. Build the wrapper:
       cd $(joinpath(@__DIR__, "cxxwrap"))
       mkdir build && cd build
       cmake .. -DCMAKE_PREFIX_PATH=$(CxxWrap.prefix_path())
       make
       make install
    """
    error("Wasserstein wrapper library not built. See instructions above.")
end

# Load the C++ module
@wrapmodule(() -> WRAPPER_PATH, :define_julia_module)

function __init__()
    @initcxx
end

# Export C++ functions
export compute_emd_cpp, compute_emds_cpp, compute_emd_from_matrix, wasserstein_version

end # module

# High-level Julia interface
using .WassersteinCxx

"""
    emd_cxx(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute EMD using the C++ Wasserstein library.

# Arguments
- `event1`: First event as (M, 3) matrix [pT, y, phi]
- `event2`: Second event as (N, 3) matrix [pT, y, phi]

# Keywords
- `R::Float64=1.0`: R parameter for EMD
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `return_flow::Bool=false`: Not supported in C++ wrapper

# Returns
- `Float64`: EMD value
"""
function emd_cxx(event1::Matrix{Float64}, event2::Matrix{Float64};
                 R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                 return_flow::Bool=false, kwargs...)

    if return_flow
        @warn "return_flow not supported in C++ wrapper, returning only EMD value"
    end

    # Handle empty events
    if size(event1, 1) == 0 || size(event2, 1) == 0
        return 0.0
    end

    # Extract weights and coordinates
    n1 = size(event1, 1)
    n2 = size(event2, 1)

    weights1 = event1[:, 1]  # pT values
    coords1 = Float64[]
    for i in 1:n1
        push!(coords1, event1[i, 2])  # y
        push!(coords1, event1[i, 3])  # phi
    end

    weights2 = event2[:, 1]
    coords2 = Float64[]
    for i in 1:n2
        push!(coords2, event2[i, 2])  # y
        push!(coords2, event2[i, 3])  # phi
    end

    # Call C++ function
    return WassersteinCxx.compute_emd_cpp(weights1, coords1, weights2, coords2, R, beta, norm)
end

"""
    emds_cxx(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix using C++ Wasserstein library.

# Arguments
- `events`: Vector of events, each as (M, 3) matrix

# Keywords
Same as `emd_cxx`

# Returns
- `Matrix{Float64}`: Symmetric matrix of pairwise EMDs
"""
function emds_cxx(events::Vector{Matrix{Float64}}; kwargs...)
    n = length(events)
    emd_matrix = zeros(Float64, n, n)

    # Extract all weights and coordinates
    weights_list = Vector{Vector{Float64}}()
    coords_list = Vector{Vector{Float64}}()

    for event in events
        if size(event, 1) == 0
            push!(weights_list, Float64[])
            push!(coords_list, Float64[])
        else
            push!(weights_list, event[:, 1])
            coords = Float64[]
            for i in 1:size(event, 1)
                push!(coords, event[i, 2])  # y
                push!(coords, event[i, 3])  # phi
            end
            push!(coords_list, coords)
        end
    end

    # Get R, beta, norm from kwargs
    R = get(kwargs, :R, 1.0)
    beta = get(kwargs, :beta, 1.0)
    norm = get(kwargs, :norm, false)

    # Call C++ batch function
    flat_matrix = WassersteinCxx.compute_emds_cpp(
        weights_list, coords_list, R, beta, norm, true
    )

    # Reshape to matrix
    for i in 1:n
        for j in 1:n
            emd_matrix[i, j] = flat_matrix[(i-1)*n + j]
        end
    end

    return emd_matrix
end

"""
    emds_cxx(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix between two sets of events.

# Arguments
- `events1`: First set of events
- `events2`: Second set of events

# Returns
- `Matrix{Float64}`: Matrix where element [i,j] is EMD(events1[i], events2[j])
"""
function emds_cxx(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)
    n1 = length(events1)
    n2 = length(events2)
    emd_matrix = zeros(Float64, n1, n2)

    for i in 1:n1
        for j in 1:n2
            emd_matrix[i, j] = emd_cxx(events1[i], events2[j]; kwargs...)
        end
    end

    return emd_matrix
end

export emd_cxx, emds_cxx