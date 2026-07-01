# EnergyFlow.jl

`EnergyFlow.jl` computes the Energy Mover's Distance (EMD) between collider events with multiple exact and approximate backends.

## Highlights

- Exact solvers with network simplex backends (`:ns64`, `:ot64`, `:ns32`, `:ot32`)
- Approximate entropic solver (`:sinkhorn`)
- Flexible ground metrics (`EuclideanMetric`, `SquaredEuclideanMetric`, `EtaPhiMetric`, `PrecomputedMetric`, `CustomMetric`)
- Single-pair and pairwise APIs (`emd`, `emds`) plus in-place workspace variants

## Installation

```julia
using Pkg
Pkg.add("EnergyFlow")
```

## Quick Start

```julia
using EnergyFlow

events = load_hepmc3_events("data/sk_example_PU.hepmc"; maxevents=20)

val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
```

Continue with the [manual](man/getting_started.md) or browse the [API reference](api.md).