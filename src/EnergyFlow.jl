"""
    EnergyFlow

A Julia package for computing the **Energy Mover's Distance (EMD)** between
collider events, with multiple solver backends and ground metrics.

The EMD ([Komiske, Metodiev, Thaler, 2019](https://arxiv.org/abs/1902.02346))
is an optimal-transport distance between events viewed as energy distributions
over a metric space:

```math
\\mathrm{EMD}_{\\beta,R}(\\mathcal{E}_0, \\mathcal{E}_1)
    = \\min_{f_{ij} \\ge 0} \\sum_{ij} f_{ij} \\left(\\frac{d_{ij}}{R}\\right)^{\\beta}
```

subject to transport constraints, where ``d_{ij}`` is the ground distance
between particles ``i`` and ``j``.

# Main entry points
- [`emd`](@ref) / [`emd!`](@ref): single-pair EMD (allocating / workspace-reusing)
- [`emds`](@ref) / [`emds!`](@ref): pairwise EMDs over collections of events
- [`set_backend`](@ref) / [`get_backend`](@ref): choose the default solver backend
- [`load_hepmc3_events`](@ref): read events from a HepMC3 ASCII file

# Event format
Events are `M × (1 + d)` matrices with one row per particle: column 1 holds
the particle weight (typically ``p_T`` or energy) and columns `2:(1+d)` hold
the ground-space coordinates (typically rapidity ``y`` and azimuth ``\\phi``).

# Backends
`:ns64` (default), `:ot64`, `:ns32`, `:ot32` (exact network simplex variants)
and `:sinkhorn` (approximate entropic OT). See [`set_backend`](@ref).

# Ground metrics
[`EuclideanMetric`](@ref) (default), [`SquaredEuclideanMetric`](@ref),
[`EtaPhiMetric`](@ref), [`PrecomputedMetric`](@ref), [`CustomMetric`](@ref).
"""
module EnergyFlow

# Core solver
include("NetworkSimplex.jl")

# Ground metric type definitions (no dependencies on EMDWorkspace)
include("GroundMetrics.jl")

# EMD workspace + single-pair solvers
include("EMDSolver.jl")

# Ground distance computation (cost filling — depends on EMDWorkspace)
include("Distances.jl")

# Event utilities and I/O
include("Utils.jl")

# Pairwise EMD
include("PairwiseEMD.jl")

# Sinkhorn solver
include("Sinkhorn.jl")

# HepMC3 event parser
include("HepMC3.jl")

# Backend dispatch (must be last)
include("EMD.jl")

# Exports (all in one place)
export NetworkSimplexSolver, network_simplex!
export EMDWorkspace
export GroundMetric, EuclideanMetric, SquaredEuclideanMetric, EtaPhiMetric, PrecomputedMetric, CustomMetric
export emd_ns64, emd_ns64!, emd_ot64, emd_ot64!
export emd_ns32, emd_ns32!, emd_ot32, emd_ot32!
export emds_ns64, emds_ns64!, emds_ot64, emds_ot64!
export emds_ns32, emds_ns32!, emds_ot32, emds_ot32!
export emd_sinkhorn, emd_sinkhorn!, emds_sinkhorn
export SinkhornWorkspace
export emd, emd!, emds, emds!
export set_backend, get_backend
export load_hepmc3_events

# PrecompileTools workload (must be last: exercises the fully-defined API).
include("Precompile.jl")

end # module
