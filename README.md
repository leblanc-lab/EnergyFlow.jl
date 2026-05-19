# EnergyFlow.jl

<!-- [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://energyflow.github.io/EnergyFlow.jl/stable)
[![Build Status](https://github.com/leblanc-lab/EnergyFlow.jl/workflows/CI/badge.svg)](https://github.com/leblanc-lab/EnergyFlow.jl/actions)
[![Coverage](https://codecov.io/gh/leblanc-lab/EnergyFlow.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/leblanc-lab/EnergyFlow.jl) -->

A Julia package for computing the **Energy Mover's Distance (EMD)** between collider events, with multiple solver backends and ground metrics, as a flexible alternative to the Python [EnergyFlow](https://energyflow.network/) package.

## Quick start

Requires Julia ≥ 1.9.
```julia
using EnergyFlow

# Load events from a HepMC3 file
events = load_hepmc3_events("data/sk_example_PU.hepmc"; maxevents=20) # read first 20 events

# or simply create with arrays of (pT, y, phi for each particle)
# event1 = [1.0 0.5 0.1;    # particle 1
#           0.8 -0.3 0.4;   # particle 2  
#           0.6 0.1 -0.2]   # particle 3

# event2 = [0.9 0.4 0.0;    # particle 1
#           0.7 -0.2 0.3]   # particle 2

# Single-pair EMD (default: :ns64, Euclidean)
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)

# Cross-pairwise: returns a 10×10 matrix
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)

# Self-pairwise: flat upper-triangular vector of length n*(n-1)/2
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
```


## Features

- Fast exact solvers: Network Simplex (`:ns64`, `:ns32`) and arc mixing Network Simplex(`:ot64`, `:ot32`)
- Approximate entropic solver: Sinkhorn (`:sinkhorn`)
- Multiple ground metrics: Euclidean, squared Euclidean, (η, φ), precomputed, or custom
- Single-pair and pairwise (self / cross) EMD/EMDs
- In-place APIs with reusable workspaces for tight loops
- HepMC3 event loading via `LorentzVectorHEP`

### Backends and metrics

```julia
# Switch backend per call…
emd(e1, e2; backend=:ot64, metric=EtaPhiMetric())

# …or globally
set_backend(:ns32)
```

| Backend     | Description                                |
|-------------|--------------------------------------------|
| `:ns64`     | Network Simplex, Float64 *(default, exact)*|
| `:ot64`     | OT-style arc mixing, Float64 *(exact)*     |
| `:ns32`     | Network Simplex, Float32                   |
| `:ot32`     | OT-style arc mixing, Float32               |
| `:sinkhorn` | Entropic OT *(approximate, fast)*          |

Available metrics: `EuclideanMetric`, `SquaredEuclideanMetric`, `EtaPhiMetric`, `PrecomputedMetric(matrix)`, `CustomMetric(f)`.

### Reusable workspace

```julia
ws = EMDWorkspace(beta=1.0, R=1.0, norm=true)
val = emd!(ws, events[1], events[2])
```

## Layout

- [src/](src/) — package source
- [example/emd/](example/emd/) — runnable EMD example ([.jl](example/emd/example_emd.jl), [.ipynb](example/emd/example_emd.ipynb), [walkthrough](example/emd/example_emd.md))
- [test/](test/) — unit tests
- [benchmark/](benchmark/) — performance benchmarks
- [data/](data/) — sample HepMC3 events
- [paper/](paper/) — for submission to JOSS

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```