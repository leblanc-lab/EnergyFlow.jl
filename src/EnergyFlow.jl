module EnergyFlow

using LinearAlgebra
using Statistics
using Distances
using JuMP
using HiGHS

# Export main EMD functions
export emd, emds, EMDParameters
export process_event, prepare_event_for_emd

# Include submodules
include("exact_solver.jl")
include("emd.jl")
include("utils.jl")
include("distances.jl")

# Package metadata
const _PACKAGE_VERSION = VersionNumber("0.1.0")

"""
    EnergyFlow

A Julia package for computing Energy Mover's Distance (EMD) and other particle physics
observables, inspired by the Python EnergyFlow package.

## Main Functions

- `emd(event1, event2; kwargs...)`: Compute EMD between two events
- `emds(events; kwargs...)`: Compute pairwise EMD matrix
- `prepare_event_for_emd(event; kwargs...)`: Prepare event data for EMD calculation

## Example

```julia
using EnergyFlow

# Create example events (pT, y, phi)
event1 = [1.0 0.5 0.1; 0.8 -0.3 0.4; 0.6 0.1 -0.2]
event2 = [0.9 0.4 0.0; 0.7 -0.2 0.3]

# Compute EMD
distance = emd(event1, event2)
```
"""
EnergyFlow

end # module