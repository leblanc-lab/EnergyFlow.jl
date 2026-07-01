# Backends and Metrics

## Solver Backends

`EnergyFlow.jl` supports exact network-simplex-style solvers and an approximate Sinkhorn solver.

| Backend     | Description                                |
|-------------|--------------------------------------------|
| `:ns64`     | Network simplex, `Float64` (default)       |
| `:ot64`     | Arc-mixing style network simplex, `Float64`|
| `:ns32`     | Network simplex, `Float32`                 |
| `:ot32`     | Arc-mixing style network simplex, `Float32`|
| `:sinkhorn` | Entropic OT (approximate)                  |

Select backend per-call:

```julia
val = emd(e1, e2; backend=:ot64)
```

Or set a global default:

```julia
set_backend(:ns32)
```

## Ground Metrics

Built-in metrics:

- `EuclideanMetric()`
- `SquaredEuclideanMetric()`
- `EtaPhiMetric()`
- `PrecomputedMetric(cost_matrix)`
- `CustomMetric(f)`

Example:

```julia
val = emd(e1, e2; metric=EtaPhiMetric(), R=1.0, beta=1.0)
```