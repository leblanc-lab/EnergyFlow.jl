# CxxSolver.jl - C++ Wasserstein solver plugin

module CxxSolver

using ..Utils

# Check if C++ wrapper library exists
const CXX_LIB_PATH = joinpath(@__DIR__, "libwasserstein_wrapper.dylib")
const CXX_LIB_EXISTS = isfile(CXX_LIB_PATH)

# For now, C++ support is disabled unless library exists
const CXX_AVAILABLE = false

if !CXX_LIB_EXISTS
    @warn "C++ Wasserstein library not found at $CXX_LIB_PATH. Build with: cd src/cxxwrap && ./build_wrapper.sh"
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

    error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
end

"""
    solve_emds(events::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD matrix using the C++ Wasserstein library.

# Returns
- `Matrix{Float64}`: Symmetric EMD distance matrix
"""
function solve_emds(events::Vector{Matrix{Float64}};
                    R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                    symmetric::Bool=true, kwargs...)

    error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
end

"""
    solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}}; kwargs...)

Compute pairwise EMD between two sets of events using C++ solver.
"""
function solve_emds(events1::Vector{Matrix{Float64}}, events2::Vector{Matrix{Float64}};
                    R::Float64=1.0, beta::Float64=1.0, norm::Bool=false,
                    kwargs...)

    error("C++ solver not available. Build with: cd src/cxxwrap && ./build_wrapper.sh")
end

# Check if the solver is available
is_available() = CXX_AVAILABLE

export solve_emd, solve_emds, is_available

end # module CxxSolver