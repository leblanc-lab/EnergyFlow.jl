<<<<<<< Updated upstream
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

# EMD functionality
include("exact_solver.jl")
include("emd.jl")
include("distances.jl")
export emd, emds, EMDParameters

# Utility functions for event processing
include("utils.jl")
export process_event, prepare_event_for_emd

# Package metadata
const _PACKAGE_VERSION = VersionNumber("0.1.0")

end # module
=======
module EnergyFlow

# Disable precompilation: @threads closures precompiled with 1 thread (Julia 1.12)
# are broken at runtime with multiple threads. Runtime JIT compilation is correct.
__precompile__(false)

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

# HepMC3 event parser (requires LorentzVectorHEP)
try
    include("HepMC3.jl")
catch
    @warn "HepMC3 submodule not loaded (LorentzVectorHEP not available)"
end

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

end # module
>>>>>>> Stashed changes
