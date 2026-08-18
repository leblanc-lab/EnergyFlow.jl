# EMD Example — EnergyFlow.jl

Files in this folder:

- [example_emd.jl](example_emd.jl) — runnable Julia script
- [example_emd.ipynb](example_emd.ipynb) — Jupyter notebook version

Run the script:

```bash
cd example/emd
julia --project=. example_emd.jl
```
Or open the notebook in Jupyter.

## 1. Load events

Events are read from a HepMC3 file. Each event becomes an `M × (1 + gdim)`
matrix in which column 1 holds the energy or `pT` weight and the remaining
columns hold the coordinates.

```julia
events = load_hepmc3_events(
    joinpath(@__DIR__, "..", "..", "data", "sk_example_PU.hepmc");
    maxevents = 20,
)
```

## 2. Single-pair EMD

`emd(ev0, ev1; ...)` returns the EMD between two events.

```julia
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
```

Key keyword arguments:

| arg       | meaning                                    | default            |
|-----------|--------------------------------------------|--------------------|
| `R`       | jet radius / cost normalization            | `1.0`              |
| `beta`    | exponent on the ground distance            | `1.0`              |
| `norm`    | normalize event energies to 1              | `false`            |
| `backend` | solver (see below)                         | `:ns64`            |
| `metric`  | ground distance (see below)                | `EuclideanMetric()`|

### Backends

| symbol      | description                                     |
|-------------|-------------------------------------------------|
| `:ns64`     | Network Simplex, Float64 *(default, exact)*     |
| `:ot64`     | OT-style arc mixing, Float64 *(exact)*          |
| `:ns32`     | Network Simplex, Float32                        |
| `:ot32`     | OT-style arc mixing, Float32                    |
| `:sinkhorn` | Sinkhorn entropic OT *(approximate, fast)*      |

Switch globally with `set_backend(:ot64)` or per-call with `backend=:ot64`.

### Ground metrics

- `EuclideanMetric()` *(default)*
- `SquaredEuclideanMetric()`
- `EtaPhiMetric()` — collider-aware (η, φ) with periodic φ
- `PrecomputedMetric(matrix)` — supply your own cost matrix
- `CustomMetric(f)` — supply a function `f(p, q) -> Real`

Example with a different backend and metric:

```julia
val2 = emd(events[1], events[2];
           R=1.0, beta=1.0, norm=true,
           backend=:ot64, metric=EtaPhiMetric())
```

## 3. Pairwise EMD

`emds(...)` computes many EMDs in one call.

**Cross-pairwise** — two event lists give an `n₀ × n₁` matrix:

```julia
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
# D[i, j] = emd(events[i], events[10+j])
```

**Self-pairwise** — one event list gives a flat upper-triangular vector of length `n*(n-1)/2`:

```julia
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
```

Pairwise calls accept `backend`, `metric`, `R`, `beta`, `norm`, `gdim`,
`n_iter_max`, and `strict`. Transport plans are available only from the
single-pair API.

## 4. Reusing workspaces

For repeated single-pair calls in a tight loop, reuse the solver buffers with
`EMDWorkspace` and `emd!`. The workspace is sized for the largest events it
will see and holds the EMD parameters (`beta`, `R`, `norm`, `metric`):

```julia
n = maximum(size(e, 1) for e in events)   # largest particle count
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
val = emd!(ws, events[1], events[2])
```

This reduces allocations by reusing the large solver buffers. The public
`emd!` path still allocates temporary arrays while unpacking each event.

For self-pairwise computations, `emds!` can reuse a caller-provided output
vector, but it manages its worker workspaces internally.

## 5. Transport plan visualisation

`emd(...; return_flow=true)` also returns the optimal transport plan matrix.
That makes it easy to draw a red/blue particle-pair plot like the one shown on
[EnergyFlow](https://energyflow.network/): blue source particles, red target
particles, and lines whose thickness follows the transported weight.

```julia
val, plan = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true, return_flow=true)
```

The runnable script in this folder saves a simple visualisation as
`emd_transport_plan.png` using `Plots.jl`.
