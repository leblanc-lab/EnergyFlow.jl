module WassersteinCxx

using CxxWrap
@wrapmodule(() -> joinpath(@__DIR__, "libwasserstein_wrapper.dylib"))

function __init__()
    @initcxx
end

# Wrapper functions for easier Julia interface
"""
    emd_cxx(event1::Matrix{Float64}, event2::Matrix{Float64}; R=1.0, beta=1.0, norm=false)

Compute EMD using C++ Wasserstein library.
"""
function emd_cxx(event1::Matrix{Float64}, event2::Matrix{Float64};
                 R::Float64=1.0, beta::Float64=1.0, norm::Bool=false)
    # Extract weights and coordinates
    weights1 = StdVector(Vector{Float64}(event1[:, 1]))
    weights2 = StdVector(Vector{Float64}(event2[:, 1]))

    # Convert coordinates to flat array format expected by C++: [y1, phi1, y2, phi2, ...]
    # event[:, 2:end] gives [y1 phi1; y2 phi2; ...], we need to flatten row-wise
    coords1 = Float64[]
    for i in 1:size(event1, 1)
        append!(coords1, event1[i, 2:end])
    end

    coords2 = Float64[]
    for i in 1:size(event2, 1)
        append!(coords2, event2[i, 2:end])
    end

    coords1_std = StdVector(coords1)
    coords2_std = StdVector(coords2)

    return compute_emd_cpp(weights1, coords1_std, weights2, coords2_std, R, beta, norm)
end

"""
    emds_cxx(events::Vector{Matrix{Float64}}; R=1.0, beta=1.0, norm=false, symmetric=true)

Compute pairwise EMD matrix using C++ Wasserstein library.
"""
function emds_cxx(events::Vector{Matrix{Float64}};
                  R::Float64=1.0, beta::Float64=1.0, norm::Bool=false, symmetric::Bool=true)
    # Convert events to C++ format
    weights_list = StdVector([StdVector(Vector{Float64}(ev[:, 1])) for ev in events])

    # Convert coordinates to flat array format expected by C++: [y1, phi1, y2, phi2, ...]
    coords_list = StdVector{StdVector{Float64}}()
    for ev in events
        coords = Float64[]
        for i in 1:size(ev, 1)
            append!(coords, ev[i, 2:end])
        end
        push!(coords_list, StdVector(coords))
    end

    # Call C++ function
    flat_matrix = compute_emds_cpp(weights_list, coords_list, R, beta, norm, symmetric)

    # Reshape to matrix
    n = length(events)
    return reshape(flat_matrix, n, n)
end

function emds_cxx(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}};
                  R::Float64=1.0, beta::Float64=1.0, norm::Bool=false)
    # For asymmetric computation, we need to handle it differently
    # Since the C++ function expects symmetric computation, we'll do it manually
    n1 = length(events1)
    n2 = length(events2)
    result = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            result[i, j] = emd_cxx(events1[i], events2[j]; R=R, beta=beta, norm=norm)
        end
    end

    return result
end

# Export the wrapped functions
export emd_cxx, emds_cxx, compute_emd_cpp, compute_emds_cpp

end # module
