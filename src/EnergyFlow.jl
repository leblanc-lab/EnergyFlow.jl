"""
    EnergyFlow

A Julia package for computing Energy Mover's Distance (EMD) and other particle physics
observables, inspired by the Python EnergyFlow package.

## Usage

This module can help you calculate EMD (Earth Mover's Distance) for comparing particle physics events. To use it, see documentation for `emd(event1, event2; kwargs...)`
This module can help you generate pairwise EMD distance matrices for analyzing multiple particle physics events simultaneously. To use it, see documentation for `emds(events; kwargs...)`

"""
module EnergyFlow

using LinearAlgebra
using Statistics
using Distances
using JuMP
using HiGHS
using LoopVectorization
using StaticArrays

# EMD functionality
include("ExactSolver.jl")
include("EMD.jl")
include("Distances.jl")
include("NetworkSimplexSolver.jl")
using .Wasserstein

# Try to load C++ wrapper, but make it optional
const CXX_WRAPPER_AVAILABLE = try
    include("WassersteinCxx.jl")
    using .WassersteinCxx
    true
catch e
    @warn "C++ Wasserstein wrapper not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh" exception=e
    false
end

export EMDParameters
export emd_exact, emds_exact  # HiGHS exact solver
export emd_network_simplex, emds_network_simplex  # Wasserstein network simplex solver
if CXX_WRAPPER_AVAILABLE
    export emd_cxx, emds_cxx  # C++ Wasserstein library
end
export euclidean_distance_2d, euclidean_distance_3d, periodic_phi_distance
export compute_distance_matrix_2d_specialized, compute_distance_matrix_3d_specialized

# Backend configuration for EMD computation
"""
Available backends for EMD computation:
- `:network_simplex` (default): Fast network simplex solver
- `:highs`: Exact solver using HiGHS
- `:cxx`: C++ Wasserstein library (if available)
"""
const AVAILABLE_BACKENDS = [:network_simplex, :highs, :cxx]

# Mutable backend setting (can be changed at runtime)
const EMD_BACKEND = Ref(:network_simplex)

"""
    set_backend(backend::Symbol)

Set the backend for EMD computation.

# Arguments
- `backend`: One of `:network_simplex` (default), `:highs`, or `:cxx`

# Examples
```julia
set_backend(:highs)  # Use exact HiGHS solver
set_backend(:network_simplex)  # Use fast network simplex (default)
set_backend(:cxx)  # Use C++ implementation (if available)
```
"""
function set_backend(backend::Symbol)
    if backend ∉ AVAILABLE_BACKENDS
        error("Invalid backend: $backend. Available backends: $AVAILABLE_BACKENDS")
    end
    if backend == :cxx && !CXX_WRAPPER_AVAILABLE
        error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
    end
    EMD_BACKEND[] = backend
    @info "EMD backend set to: $backend"
end

"""
    get_backend()

Get the current EMD computation backend.
"""
get_backend() = EMD_BACKEND[]

# Dynamic dispatch functions that use the selected backend
"""
    emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute the Earth Mover's Distance (EMD) between two events using the selected backend.

# Arguments
- `event1`: First event as (M, 3) matrix [pT, y, phi]
- `event2`: Second event as (N, 3) matrix [pT, y, phi]

# Keywords
See backend-specific functions for documentation of parameters.

# Examples
```julia
event1 = [1.0 0.0 0.0; 1.0 0.5 0.5]
event2 = [1.0 0.1 0.1; 1.0 0.6 0.6]
emd_value = emd(event1, event2)
```
"""
function emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)
    backend = EMD_BACKEND[]
    if backend == :network_simplex
        return emd_network_simplex(event1, event2; kwargs...)
    elseif backend == :highs
        return emd_exact(event1, event2; kwargs...)
    elseif backend == :cxx
        if !CXX_WRAPPER_AVAILABLE
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        return emd_cxx(event1, event2; kwargs...)
    else
        error("Unknown backend: $backend")
    end
end

"""
    emds(events::Vector{Matrix{Float64}}; kwargs...)
    emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix using the selected backend.

# Arguments
- `events`: Vector of events for symmetric pairwise computation
- `events1`, `events2`: Two sets of events for asymmetric computation

# Keywords
See backend-specific functions for documentation of parameters.

# Examples
```julia
events = [event1, event2, event3]
emd_matrix = emds(events)
```
"""
function emds(events::Vector{Matrix{Float64}}; kwargs...)
    backend = EMD_BACKEND[]
    if backend == :network_simplex
        return emds_network_simplex(events; kwargs...)
    elseif backend == :highs
        return emds_exact(events; kwargs...)
    elseif backend == :cxx
        if !CXX_WRAPPER_AVAILABLE
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        return emds_cxx(events; kwargs...)
    else
        error("Unknown backend: $backend")
    end
end

function emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)
    backend = EMD_BACKEND[]
    if backend == :network_simplex
        return emds_network_simplex(events1, events2; kwargs...)
    elseif backend == :highs
        return emds_exact(events1, events2; kwargs...)
    elseif backend == :cxx
        if !CXX_WRAPPER_AVAILABLE
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        return emds_cxx(events1, events2; kwargs...)
    else
        error("Unknown backend: $backend")
    end
end

export emd, emds, set_backend, get_backend

# Utility functions for event processing
include("Utils.jl")
export process_event, prepare_event_for_emd

# Package metadata
const _PACKAGE_VERSION = VersionNumber("0.1.0")

end # module