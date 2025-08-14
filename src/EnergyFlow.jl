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