# Backends and Metrics

## Solver Backends

`EnergyFlow.jl` ships several solver backends. The network-simplex variants
target the exact linear-program optimum; the Sinkhorn backend solves an
entropy-regularized approximation and can be faster for large events.

| Backend     | Solver                             | Precision | Exact? |
|:------------|:-----------------------------------|:----------|:-------|
| `:ns64`     | Network simplex *(default)*        | `Float64` | yes    |
| `:ot64`     | Network simplex with arc mixing    | `Float64` | yes    |
| `:ns32`     | Network simplex                    | `Float32` | yes    |
| `:ot32`     | Network simplex with arc mixing    | `Float32` | yes    |
| `:sinkhorn` | Entropy-regularised OT (Sinkhorn)  | `Float64` | no     |

### Selecting a backend

Per call:

```julia
val = emd(ev0, ev1; backend=:ot64)
val = emd(ev0, ev1; backend=EnergyFlow.OT64)  # typed equivalent
```

Or globally, for all subsequent calls that don't specify one:

```julia
set_backend(:ot64)
set_backend(EnergyFlow.OT64)  # typed equivalent
get_backend()   # :ot64
```

The symbols are retained as a compatibility interface. Internally they are
converted once to the singleton strategies `EnergyFlow.NS64`,
`EnergyFlow.OT64`, `EnergyFlow.NS32`, `EnergyFlow.OT32`, and
`EnergyFlow.Sinkhorn`, after which Julia dispatch selects the implementation.

Each backend is also available as a directly callable function
(`emd_ns64`, `emd_ot64`, `emd_ns32`, `emd_ot32`, `emd_sinkhorn`, and the
pairwise/in-place variants) — see the [API Reference](../api.md).

### Which backend should I use?

- **`:ns64`** (the default) is a robust exact solver — a good starting point
  for most workloads.
- **`:ot64`** uses POT-style *arc mixing*: bipartite arcs are interleaved so
  that each pivot block samples arcs from many source particles. This can
  improve convergence for unbalanced problems
  (`norm=false` with very different event weights).
- **`:ns32` / `:ot32`** trade precision for memory and speed. The cost matrix
  dominates memory usage (`n0 × n1` values), so halving the element size can
  matter for large events. EMD values are returned as `Float32`.
- **`:sinkhorn`** solves the entropy-regularized problem. The result carries a
  bias controlled by `epsilon` (smaller `epsilon` → closer to exact, but
  typically more iterations). Use it when approximate distances suffice.

```julia
# Sinkhorn with a tighter regularisation
val = emd(ev0, ev1; backend=:sinkhorn)
val = emd_sinkhorn(ev0, ev1; epsilon=0.005, annealing=true)
```

!!! note
    `emds!` (in-place pairwise) is not available for the `:sinkhorn` backend;
    use `emds` instead.

### Convergence

All solvers have iteration limits. The exact backends warn and return `NaN`
when they stop without an optimal solution. Sinkhorn may return a finite last
iterate that has not met its marginal tolerance. Pass `strict=true` to `emd`
or `emds` when a non-optimal status must be an error. After a workspace-based
Sinkhorn solve, inspect `ws.converged` and `ws.n_iters` for diagnostics.

## Ground Metrics

The ground metric defines the distance ``d_{ij}`` between particles, from
which arc costs ``(d_{ij}/R)^\beta`` are built. Select it with the `metric`
keyword:

```julia
val = emd(ev0, ev1; metric=EtaPhiMetric(), R=1.0, beta=1.0)
```

### [`EuclideanMetric`](@ref) *(default)*

Plain Euclidean distance over all coordinate dimensions:

```math
d_{ij} = \sqrt{\textstyle\sum_k (\Delta x_k)^2}
```

### [`SquaredEuclideanMetric`](@ref)

Squared Euclidean distance (no square root), i.e.
``d_{ij} = \sum_k (\Delta x_k)^2``. Equivalent to `EuclideanMetric` with
`beta = 2` when `R = 1`; useful for Wasserstein-2-style costs.

### [`EtaPhiMetric`](@ref)

Collider-aware ``(\eta, \phi)`` distance with periodic azimuth:

```math
d_{ij} = \sqrt{\Delta\eta^2 + \Delta\phi_{\mathrm{wrap}}^2}, \qquad
\Delta\phi_{\mathrm{wrap}} = \pi - \bigl|\,\mathrm{mod}(|\Delta\phi|, 2\pi) - \pi\,\bigr|
```

Use this for events in `[pT, η, φ]` format whenever particles can lie near
the ``\phi = \pm\pi`` boundary — the plain Euclidean metric would overestimate
those distances. Assumes coordinate 1 is ``\eta`` and coordinate 2 is ``\phi``.

### [`PrecomputedMetric`](@ref)

Supply the full cost matrix yourself. Values are used **as-is**; `R` and
`beta` are *not* applied:

```julia
C = [my_cost(i, j) for i in 1:size(ev0, 1), j in 1:size(ev1, 1)]
val = emd(ev0, ev1; metric=PrecomputedMetric(C))
```

### [`CustomMetric`](@ref)

Supply a distance function `f(p, q) -> Real` taking views of two particles'
coordinate vectors. `R` and `beta` *are* applied to the result:

```julia
manhattan = CustomMetric((p, q) -> sum(abs, p .- q))
val = emd(ev0, ev1; metric=manhattan)
```

!!! tip
    The built-in metrics (`EuclideanMetric`, `SquaredEuclideanMetric`,
    `EtaPhiMetric`) have specialized, SIMD-friendly cost-fill loops and
    support parallel filling for large events. `PrecomputedMetric` and
    `CustomMetric` always fill serially.

!!! note
    The `:sinkhorn` backend currently supports only `EuclideanMetric()`.
    Passing another metric through `emd` or `emds` raises `ArgumentError`.
