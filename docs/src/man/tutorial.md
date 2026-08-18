# Tutorial: Comparing Collider Events

This tutorial walks through a complete EMD analysis using the sample data
shipped with the package. A runnable version lives in the repository at
`example/emd/` as both a script (`example_emd.jl`) and a Jupyter notebook
(`example_emd.ipynb`):

```bash
cd example/emd
julia --project=. example_emd.jl
```

## 1. Load events

The repository ships a sample HepMC3 file with pileup events,
`data/sk_example_PU.hepmc`. [`load_hepmc3_events`](@ref) parses it into
event matrices — one `M × 3` matrix of `[pT, η, φ]` rows per event, keeping
final-state particles only:

```julia
using EnergyFlow

datafile = joinpath(pkgdir(EnergyFlow), "data", "sk_example_PU.hepmc")
events = load_hepmc3_events(datafile; maxevents=20)

println("Loaded $(length(events)) events")
println("Event 1 has $(size(events[1], 1)) particles")
```

Each row of an event is a particle: column 1 is its ``p_T`` (the transported
"mass"), columns 2–3 are its position in the ``(\eta, \phi)`` plane.

## 2. A first EMD

Compute the distance between the first two events:

```julia
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
println("EMD(event 1, event 2) = $val")
```

With `norm=true` both events are rescaled to unit total weight, so `val`
measures how differently the two events *distribute* their energy, on a scale
set by `R`. Try `norm=false` to also account for the difference in total
``p_T`` (created/destroyed energy costs 1 per unit).

## 3. Backends and metrics

The default is an exact network-simplex solver in `Float64` (`:ns64`).
You can switch solvers and ground metrics per call:

```julia
# OT-style arc-mixing solver + collider-aware (η, φ) metric
val2 = emd(events[1], events[2];
           R=1.0, beta=1.0, norm=true,
           backend=:ot64, metric=EtaPhiMetric())
```

[`EtaPhiMetric`](@ref) wraps the azimuthal angle, so two particles at
``\phi = 3.1`` and ``\phi = -3.1`` are treated as close — the plain Euclidean
metric would put them ``\approx 2\pi`` apart.

Other options (see [Backends and Metrics](backends_and_metrics.md)):

```julia
emd(events[1], events[2]; backend=:ns32)      # Float32 precision
emd(events[1], events[2]; backend=:sinkhorn)  # fast approximation
set_backend(:ot64)                            # change the global default
```

## 4. Pairwise distances

Most analyses need distances between *many* events. [`emds`](@ref) does this
in one multithreaded call.

Cross-pairwise — compare one set of events against another:

```julia
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
size(D)     # (10, 10);  D[i, j] = emd(events[i], events[10 + j])
```

Self-pairwise — all distinct pairs within one set, returned as a flat
upper-triangular vector (SciPy `pdist` order):

```julia
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
length(dists)   # 45 == 10*9/2
```

This flat vector is exactly what clustering packages such as
[Clustering.jl](https://juliastats.org/Clustering.jl/stable/) expect for
hierarchical clustering, e.g. building a dendrogram of events:

```julia
using Clustering
n = 10
D = zeros(n, n)
k = 1
for i in 1:n, j in (i+1):n
    D[i, j] = D[j, i] = dists[k]
    global k += 1
end
tree = hclust(D; linkage=:average)
```

## 5. Fast repeated computation with workspaces

`emd` allocates fresh buffers on every call. When computing EMDs in a hot
loop — scans over parameters, nearest-neighbour searches, custom pairings —
pre-allocate an [`EMDWorkspace`](@ref) sized for your largest events and use
[`emd!`](@ref):

```julia
n = maximum(size(e, 1) for e in events)
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)

# Distances from event 1 to all others, reusing the solver workspace
d1 = [emd!(ws, events[1], e) for e in events[2:end]]
```

The EMD parameters are fixed in the workspace; create separate workspaces for
different parameter settings. The public `emd!` path still allocates temporary
arrays while unpacking each event, but reuses the much larger solver buffers.
For the Sinkhorn backend use a [`SinkhornWorkspace`](@ref) instead.

## 6. Threading

Start Julia with threads to parallelize pairwise computations:

```bash
julia -t auto
```

`emds`/`emds!` distribute event pairs across threads (each thread keeps its
own internal workspace), and single-pair calls parallelize the cost-matrix
fill for very large events.

## Full script

The complete example, ready to run:

```julia
using EnergyFlow

# Load events from HepMC3
events = load_hepmc3_events(
    joinpath(pkgdir(EnergyFlow), "data", "sk_example_PU.hepmc");
    maxevents=20,
)
println("Loaded $(length(events)) events")

# ── Single EMD ──
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
println("EMD (ns64, Euclidean): $val")

val2 = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true,
           backend=:ot64, metric=EtaPhiMetric())
println("EMD (ot64, EtaPhi):    $val2")

# ── Pairwise EMD ──
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
println("Cross-pairwise (10×10): D[1,1] = $(D[1,1])")

dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
println("Self-pairwise ($(length(dists)) pairs): dists[1] = $(dists[1])")

# ── Workspace reuse ──
n = maximum(size(e, 1) for e in events)
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
val3 = emd!(ws, events[1], events[2])
println("EMD via workspace:     $val3")
```
