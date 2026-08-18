# EnergyFlow.jl

*Energy Mover's Distance for collider events in Julia.*

`EnergyFlow.jl` computes the **Energy Mover's Distance (EMD)** between
collider events, with multiple exact and approximate solver backends and a
flexible choice of ground metrics. It is a Julia alternative to the Python
[EnergyFlow](https://energyflow.network/) package.

The EMD ([Komiske, Metodiev, Thaler, 2019](https://arxiv.org/abs/1902.02346))
views each event as a distribution of energy over a metric space and measures
the minimum "work" — energy times distance — needed to rearrange one event
into the other. Its balanced transport term is

```math
\mathrm{EMD}_{\beta,R}(\mathcal{E}_0, \mathcal{E}_1)
    = \min_{f_{ij} \ge 0} \sum_{ij} f_{ij} \left(\frac{d_{ij}}{R}\right)^{\beta}
```

subject to transport constraints, where ``f_{ij}`` is the energy transported
from particle ``i`` to particle ``j`` and ``d_{ij}`` is their ground distance.
For unnormalized events with unequal total weight, EnergyFlow.jl adds a
unit-cost fictitious particle to balance the transport problem.

## Highlights

- **Exact solvers** based on the network simplex algorithm
  (`:ns64`, `:ot64`, `:ns32`, `:ot32`)
- **Approximate entropic solver** (`:sinkhorn`) for speed at scale
- **Flexible ground metrics**: [`EuclideanMetric`](@ref),
  [`SquaredEuclideanMetric`](@ref), [`EtaPhiMetric`](@ref),
  [`PrecomputedMetric`](@ref), [`CustomMetric`](@ref)
- **Single-pair and pairwise APIs** ([`emd`](@ref), [`emds`](@ref)), with
  multithreaded pairwise computation
- **Allocation-reduced APIs**: reusable workspaces for [`emd!`](@ref) and
  preallocated result storage for [`emds!`](@ref)
- **HepMC3 event loading** with [`load_hepmc3_events`](@ref)

## Installation

Requires Julia 1.9 or later. Install EnergyFlow.jl from Julia's General registry:

```julia
using Pkg
Pkg.add("EnergyFlow")
```

## Quick Start

```julia
using EnergyFlow

# Events are M×3 matrices: one row per particle, columns [pT, y, φ]
ev0 = [1.0  0.5  0.1;
       0.8 -0.3  0.4;
       0.6  0.1 -0.2]

ev1 = [0.9  0.4  0.0;
       0.7 -0.2  0.3]

# Single-pair EMD. EtaPhiMetric wraps the azimuthal coordinate.
val = emd(ev0, ev1;
          R=1.0, beta=1.0, norm=true, metric=EtaPhiMetric())

# Cross-pairwise: a 1×1 matrix
D = emds([ev0], [ev1]; norm=true, metric=EtaPhiMetric())

# Self-pairwise: flat upper-triangular vector of length n*(n-1)/2
dists = emds([ev0, ev1]; norm=true, metric=EtaPhiMetric())
```

## Where to go next

- [Getting Started](man/getting_started.md) — event format, parameters, and
  basic usage
- [Tutorial](man/tutorial.md) — a complete worked example with real events
- [Backends and Metrics](man/backends_and_metrics.md) — choosing a solver and
  ground metric
- [API Reference](api.md) — every exported type and function

## Citing

The EMD was introduced in:

> P. T. Komiske, E. M. Metodiev, J. Thaler,
> *The Metric Space of Collider Events*,
> [Phys. Rev. Lett. 123, 041801 (2019)](https://doi.org/10.1103/PhysRevLett.123.041801),
> [arXiv:1902.02346](https://arxiv.org/abs/1902.02346).

See the repository's
[`CITATION.cff`](https://github.com/leblanc-lab/EnergyFlow.jl/blob/main/CITATION.cff)
for how to cite this package.
