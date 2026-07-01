# Getting Started

## Basic Usage

```julia
using EnergyFlow

events = load_hepmc3_events("data/sk_example_PU.hepmc"; maxevents=20)

# Single-pair EMD
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)

# Cross-pairwise distances (matrix)
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)

# Self-pairwise distances (flattened upper triangle)
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
```

## Reusing Workspaces

For repeated computations, reusing a workspace avoids repeated allocations.

```julia
ws = EMDWorkspace(beta=1.0, R=1.0, norm=true)
val = emd!(ws, events[1], events[2])
```