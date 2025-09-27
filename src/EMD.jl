# EMD.jl - Main EMD orchestrator with backend system

module EMD

using ..Utils

# Include solver modules
include("NetworkSimplexSolver.jl")
include("ExactSolver.jl")
include("CxxSolver.jl")

using .Wasserstein  # NetworkSimplexSolver exports module Wasserstein
using .ExactSolver
using .CxxSolver

# Backend configuration
"""
Available backends for EMD computation:
- `:network_simplex` (default): Fast network simplex solver
- `:exact` or `:highs`: Exact solver using HiGHS
- `:cxx`: C++ Wasserstein library (if available)
"""
const AVAILABLE_BACKENDS = [:network_simplex, :exact, :highs, :cxx]

# Mutable backend setting (can be changed at runtime)
const EMD_BACKEND = Ref(:network_simplex)

"""
    set_backend(backend::Symbol)

Set the backend for EMD computation.

# Arguments
- `backend`: One of `:network_simplex` (default), `:exact`/`:highs`, or `:cxx`

# Examples
```julia
EMD.set_backend(:exact)  # Use exact HiGHS solver
EMD.set_backend(:network_simplex)  # Use fast network simplex (default)
EMD.set_backend(:cxx)  # Use C++ implementation (if available)
```
"""
function set_backend(backend::Symbol)
    # Handle aliases
    if backend == :highs
        backend = :exact
    end

    if backend ∉ AVAILABLE_BACKENDS
        error("Invalid backend: $backend. Available backends: $AVAILABLE_BACKENDS")
    end

    if backend == :cxx && !CxxSolver.is_available()
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

"""
    emd(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute the Earth Mover's Distance (EMD) between two events using the selected backend.

# Arguments
- `event1`: First event as (N, 3+) matrix [weight, y, phi, ...]
- `event2`: Second event as (M, 3+) matrix [weight, y, phi, ...]

# Keywords
- `backend::Symbol=get_backend()`: Override backend for this call
- `R::Float64=1.0`: Maximum distance parameter
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `return_flow::Bool=false`: Whether to return flow matrix
- Backend-specific parameters...

# Returns
- EMD value (Float64) by default
- `(emd_value, flow_matrix)` if `return_flow=true`

# Examples
```julia
event1 = [1.0 0.0 0.0; 1.0 0.5 0.5]  # 2 particles: [weight y phi]
event2 = [1.0 0.1 0.1; 1.0 0.6 0.6]
emd_value = emd(event1, event2)

# Use specific backend
emd_value = emd(event1, event2, backend=:exact)
```
"""
function emd(event1::Matrix{Float64}, event2::Matrix{Float64};
             backend::Symbol=EMD_BACKEND[], return_flow::Bool=false,
             kwargs...)

    # Handle backend aliases
    if backend == :highs
        backend = :exact
    end

    # Select solver based on backend
    if backend == :network_simplex
        emd_val, status, flow = Wasserstein.solve_emd(
            event1, event2; return_flow=return_flow, kwargs...
        )
    elseif backend == :exact
        emd_val, status, flow = ExactSolver.solve_emd(
            event1, event2; return_flow=return_flow, kwargs...
        )
    elseif backend == :cxx
        if !CxxSolver.is_available()
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        emd_val, status, flow = CxxSolver.solve_emd(
            event1, event2; kwargs...
        )
    else
        error("Unknown backend: $backend")
    end

    # Check status
    if status != :success
        @warn "EMD computation returned status: $status"
    end

    # Return based on options
    if return_flow && flow !== nothing
        return emd_val, flow
    else
        return emd_val
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
- `backend::Symbol=get_backend()`: Override backend for this call
- Other parameters passed to the backend

# Examples
```julia
events = [event1, event2, event3]
emd_matrix = emds(events)

# Asymmetric computation
emd_matrix = emds(events_set1, events_set2)
```
"""
function emds(events::Vector{Matrix{Float64}};
              backend::Symbol=EMD_BACKEND[], kwargs...)

    # Handle backend aliases
    if backend == :highs
        backend = :exact
    end

    # Select solver based on backend
    if backend == :network_simplex
        return Wasserstein.solve_emds(events; kwargs...)
    elseif backend == :exact
        return ExactSolver.solve_emds(events; kwargs...)
    elseif backend == :cxx
        if !CxxSolver.is_available()
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        return CxxSolver.solve_emds(events; kwargs...)
    else
        error("Unknown backend: $backend")
    end
end

function emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}};
              backend::Symbol=EMD_BACKEND[], kwargs...)

    # Handle backend aliases
    if backend == :highs
        backend = :exact
    end

    # Select solver based on backend
    if backend == :network_simplex
        return Wasserstein.solve_emds(events1, events2; kwargs...)
    elseif backend == :exact
        return ExactSolver.solve_emds(events1, events2; kwargs...)
    elseif backend == :cxx
        if !CxxSolver.is_available()
            error("C++ backend not available. Build it with: cd src/cxxwrap && ./build_wrapper.sh")
        end
        return CxxSolver.solve_emds(events1, events2; kwargs...)
    else
        error("Unknown backend: $backend")
    end
end

# Direct access to specific solvers (for advanced users)
"""
    emd_network_simplex(event1, event2; kwargs...)

Directly use the network simplex solver.
"""
function emd_network_simplex(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)
    emd_val, _, _ = Wasserstein.solve_emd(event1, event2; kwargs...)
    return emd_val
end

"""
    emds_network_simplex(events; kwargs...)

Directly use the network simplex solver for pairwise computation.
"""
emds_network_simplex(events::Vector{Matrix{Float64}}; kwargs...) =
    Wasserstein.solve_emds(events; kwargs...)

emds_network_simplex(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...) =
    Wasserstein.solve_emds(events1, events2; kwargs...)

"""
    emd_exact(event1, event2; kwargs...)

Directly use the exact solver.
"""
function emd_exact(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)
    emd_val, _, _ = ExactSolver.solve_emd(event1, event2; kwargs...)
    return emd_val
end

"""
    emds_exact(events; kwargs...)

Directly use the exact solver for pairwise computation.
"""
emds_exact(events::Vector{Matrix{Float64}}; kwargs...) =
    ExactSolver.solve_emds(events; kwargs...)

emds_exact(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...) =
    ExactSolver.solve_emds(events1, events2; kwargs...)

"""
    emd_cxx(event1, event2; kwargs...)

Directly use the C++ solver (if available).
"""
function emd_cxx(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)
    if !CxxSolver.is_available()
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end
    emd_val, _, _ = CxxSolver.solve_emd(event1, event2; kwargs...)
    return emd_val
end

"""
    emds_cxx(events; kwargs...)

Directly use the C++ solver for pairwise computation (if available).
"""
function emds_cxx(events::Vector{Matrix{Float64}}; kwargs...)
    if !CxxSolver.is_available()
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end
    return CxxSolver.solve_emds(events; kwargs...)
end

function emds_cxx(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)
    if !CxxSolver.is_available()
        error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
    end
    return CxxSolver.solve_emds(events1, events2; kwargs...)
end

# Export main functions
export emd, emds
export set_backend, get_backend
export emd_network_simplex, emds_network_simplex
export emd_exact, emds_exact
export emd_cxx, emds_cxx

end # module EMD
