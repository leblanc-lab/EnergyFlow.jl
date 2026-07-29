# Getting Started

## Installation

Requires Julia ≥ 1.9.

```julia
using Pkg
Pkg.add("EnergyFlow")
```

For multithreaded pairwise computations, start Julia with multiple threads:

```bash
julia -t auto
```

## The event format

An event is an `M × (1 + d)` matrix with one row per particle:

- **Column 1**: the particle weight — typically transverse momentum ``p_T``
  or energy. This is the "mass" being transported.
- **Columns `2:(1+d)`**: the ground-space coordinates — typically rapidity
  ``y`` (or pseudorapidity ``\eta``) and azimuthal angle ``\phi``.

```julia
#        pT     y     φ
ev0 = [ 1.0   0.5   0.1;    # particle 1
        0.8  -0.3   0.4;    # particle 2
        0.6   0.1  -0.2 ]   # particle 3
```

Any number of coordinate dimensions is supported; the `gdim` keyword restricts
the computation to the first `gdim` coordinate columns if you have extra
columns you want ignored.

## Loading events from HepMC3 files

[`load_hepmc3_events`](@ref) reads a HepMC3 ASCII file and returns a vector of
event matrices in `[pT, η, φ]` format, keeping only final-state particles
(status code 1 by default):

```julia
using EnergyFlow

events = load_hepmc3_events("data/sk_example_PU.hepmc"; maxevents=20)
```

Useful keywords: `maxevents` (limit the number of events read), `skipevents`
(skip events at the start of the file), `status` (particle status code to
keep), and `min_pt` (drop soft particles).

## Computing a single EMD

```julia
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
```

### The parameters `R`, `beta`, and `norm`

The EMD between events ``\mathcal{E}_0`` and ``\mathcal{E}_1`` is

```math
\mathrm{EMD}_{\beta,R}(\mathcal{E}_0, \mathcal{E}_1)
    = \min_{f_{ij} \ge 0} \sum_{ij} f_{ij} \left(\frac{d_{ij}}{R}\right)^{\beta}
```

- **`R`** sets the distance scale: ground distances are divided by `R` before
  exponentiation. When `norm=false`, `R` controls the trade-off between
  *transporting* energy (cost ``(d/R)^\beta`` per unit) and
  *creating/destroying* it (cost 1 per unit via the fictitious particle).
  For the EMD to be a true metric on unnormalized events, choose
  ``R \geq d_{\max}/2`` — e.g. `R = 0.4` is a common choice for jets with
  radius parameter 0.4.
- **`beta`** is the angular exponent on the ground distance. `beta=1` gives
  the standard EMD (Wasserstein-1); `beta=2` weights large distances more
  heavily.
- **`norm`** — if `true`, each event's weights are normalized to sum to 1
  before solving, so only the *shapes* of the events are compared. If
  `false` (the default), the difference in total weight is created/destroyed
  at unit cost, matching the Python EnergyFlow convention.

## Pairwise EMDs

[`emds`](@ref) computes many EMDs in one multithreaded call.

**Cross-pairwise** — two event lists give a distance matrix:

```julia
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
# D[i, j] = emd(events[i], events[10 + j])
```

**Self-pairwise** — one event list gives the strict upper triangle of the
distance matrix as a flat vector of length `n*(n-1)/2`, in SciPy `pdist`
order (`(1,2), (1,3), …, (1,n), (2,3), …`):

```julia
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
```

To rebuild the full symmetric matrix from the flat vector:

```julia
n = 10
D = zeros(n, n)
k = 1
for i in 1:n, j in (i+1):n
    D[i, j] = D[j, i] = dists[k]
    global k += 1
end
```

## Reusing workspaces

Each plain `emd` call allocates internal buffers. For tight loops, allocate an
[`EMDWorkspace`](@ref) once — sized for the largest events you will compare —
and use [`emd!`](@ref):

```julia
n = maximum(size(e, 1) for e in events)   # largest particle count
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)

val = emd!(ws, events[1], events[2])
```

Note that with the in-place API the EMD parameters (`beta`, `R`, `norm`,
`metric`) live in the workspace, not in the call.

For pairwise computations, [`emds!`](@ref) writes into a pre-allocated result
vector (workspaces are managed internally, one per thread):

```julia
n = length(events)
results = Vector{Float64}(undef, n * (n - 1) ÷ 2)
emds!(results, events; R=1.0, beta=1.0, norm=true)
```

## Multithreading

Pairwise computations (`emds`, `emds!`) parallelize over event pairs, and
single-pair computations parallelize the cost-matrix fill for large events.
Both require Julia to be started with threads:

```bash
julia -t auto            # or e.g. -t 8
# or: export JULIA_NUM_THREADS=8
```

## Next steps

- Follow the [Tutorial](tutorial.md) for a complete worked example.
- See [Backends and Metrics](backends_and_metrics.md) for choosing solvers
  and ground distances.
