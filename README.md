# EnergyFlow.jl

[![CI](https://github.com/leblanc-lab/EnergyFlow.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/leblanc-lab/EnergyFlow.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://leblanc-lab.github.io/EnergyFlow.jl/dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

EnergyFlow.jl computes the **Energy Mover's Distance (EMD)** between collider
events. It provides exact and approximate optimal-transport solvers, several
ground metrics, pairwise computations, and event-isotropy utilities in Julia.

## Installation

EnergyFlow.jl requires Julia 1.9 or later. Until the package is registered in
Julia's General registry, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/leblanc-lab/EnergyFlow.jl")
```

## Quick start

An event is a matrix with one particle per row. The first column contains the
particle weight (usually transverse momentum or energy); the remaining columns
contain its coordinates.

```julia
using EnergyFlow

# Two events in [pT, y, phi] format
event1 = [1.0  0.5  0.1;
          0.8 -0.3  0.4;
          0.6  0.1 -0.2]

event2 = [0.9  0.4  0.0;
          0.7 -0.2  0.3]

# Single-pair EMD. EtaPhiMetric wraps the azimuthal coordinate.
distance = emd(event1, event2;
               R=1.0, beta=1.0, norm=true, metric=EtaPhiMetric())

# Cross-pairwise: a length(events_a) x length(events_b) matrix
D = emds([event1], [event2]; norm=true, metric=EtaPhiMetric())

# Self-pairwise: the strict upper triangle in SciPy pdist order
dists = emds([event1, event2]; norm=true, metric=EtaPhiMetric())
```

Use [`load_hepmc3_events`](https://leblanc-lab.github.io/EnergyFlow.jl/dev/api/#EnergyFlow.load_hepmc3_events)
to read HepMC3 ASCII files as `[pT, eta, phi]` event matrices.

## Solvers and metrics

The default backend is the exact `Float64` network-simplex solver, `:ns64`.

| Backend | Description |
|:--|:--|
| `:ns64` | Network simplex, `Float64` (default, exact) |
| `:ot64` | Network simplex with arc mixing, `Float64` (exact) |
| `:ns32` | Network simplex, `Float32` (exact) |
| `:ot32` | Network simplex with arc mixing, `Float32` (exact) |
| `:sinkhorn` | Entropy-regularized optimal transport, `Float64` (approximate) |

Choose a backend per call with `backend=:ot64`, or change the process-wide
default with `set_backend(:ot64)`. Available ground metrics are
`EuclideanMetric()`, `SquaredEuclideanMetric()`, `EtaPhiMetric()`,
`PrecomputedMetric(costs)`, and `CustomMetric(f)`.

For repeated single-pair computations, `EMDWorkspace` and `emd!` reuse the
solver's large internal buffers and reduce allocations:

```julia
workspace = EMDWorkspace(3, 2;
                         R=1.0, beta=1.0, norm=true,
                         metric=EtaPhiMetric())
distance = emd!(workspace, event1, event2)
```

See the [documentation](https://leblanc-lab.github.io/EnergyFlow.jl/dev/) for
parameter definitions, Sinkhorn controls, transport plans, event isotropy, and
the complete API.

## Development

- [`example/emd/`](example/emd/) contains a runnable EMD example and notebook.
- [`example/isotropy/`](example/isotropy/) demonstrates ring, cylinder, and
  spherical event isotropy.
- [`benchmark/`](benchmark/) contains reproducible Julia/POT comparisons.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) explains the development workflow.

Run the test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Citation and license

If you use EnergyFlow.jl in research, cite the software using
[`CITATION.cff`](CITATION.cff) and cite the original EMD paper listed in the
[documentation](https://leblanc-lab.github.io/EnergyFlow.jl/dev/).

EnergyFlow.jl is released under the [MIT License](LICENSE).
