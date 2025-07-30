# EnergyFlow.jl

<!-- [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://energyflow.github.io/EnergyFlow.jl/stable)
[![Build Status](https://github.com/energyflow/EnergyFlow.jl/workflows/CI/badge.svg)](https://github.com/energyflow/EnergyFlow.jl/actions)
[![Coverage](https://codecov.io/gh/energyflow/EnergyFlow.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/energyflow/EnergyFlow.jl) -->

A Julia package for computing Energy Mover's Distance (EMD) and other particle physics observables, inspired by the Python [EnergyFlow](https://energyflow.network/) package.

<!-- ## Installation

```julia
using Pkg
Pkg.add("EnergyFlow")
```

For the exact LP solver (recommended for best accuracy):

```julia
Pkg.add(["JuMP", "HiGHS"])
``` -->

## Quick Start

```julia
using EnergyFlow

# Create two events (pT, y, phi for each particle)
event1 = [1.0 0.5 0.1;    # particle 1
          0.8 -0.3 0.4;   # particle 2  
          0.6 0.1 -0.2]   # particle 3

event2 = [0.9 0.4 0.0;    # particle 1
          0.7 -0.2 0.3]   # particle 2

# Compute EMD
distance = emd(event1, event2)
println("EMD between events: ", distance)
```

## Key Functions

### EMD Computation

```julia
# Basic EMD
distance = emd(event1, event2)

# EMD with normalization (treats events as probability distributions)
distance = emd(event1, event2, norm=true)

# EMD with custom parameters
distance = emd(event1, event2,
    R = 1.0,              # Angular scale parameter
    beta = 1.0,           # Distance exponent
    measure = "euclidean", # or "spherical"
    periodic_phi = true    # Handle phi periodicity
)

# Get optimal transport flow matrix
distance, flow = emd(event1, event2, return_flow=true)

# Compute pairwise EMDs for multiple events
events = [event1, event2, event3, event4]
emd_matrix = emds(events)
```

### Utilities

```julia
# Apply kinematic cuts
cut_event = EnergyFlow.apply_kinematic_cuts(event, pT_min=0.5, y_max=2.5)

# Coordinate transformations
cartesian = EnergyFlow.hadronic_to_cartesian(hadronic_event)
hadronic = EnergyFlow.cartesian_to_hadronic(cartesian_event)

# Event properties
total_p = EnergyFlow.total_momentum(event)
center = EnergyFlow.center_of_energy(event)
```