#=
    EMD.jl

    EMD backend orchestrator.

    Provides the user-facing emd / emd! / emds / emds! functions that dispatch
    to the active backend. Currently implemented backends:

        :ns64       — Network Simplex, Float64  (EMDSolver.jl)
        :ot64       — OT-style (arc mixing), Float64  (EMDSolver.jl)
        :ns32       — Network Simplex, Float32  (EMDSolver.jl)
        :ot32       — OT-style (arc mixing), Float32  (EMDSolver.jl)
        :sinkhorn   — Sinkhorn regularised OT (Sinkhorn.jl)

=#

# ─────────────────────────────────────────────────────────────────────
# Backend registry
# ─────────────────────────────────────────────────────────────────────

const EMD_BACKEND = Ref(:ns64)

"""
    set_backend(backend::Union{Symbol,EMDBackend}) -> Symbol

Set the global default EMD solver backend used by [`emd`](@ref), [`emd!`](@ref),
[`emds`](@ref) and [`emds!`](@ref) when no `backend` keyword is given.

Available backends:

| Backend     | Solver                              | Precision | Exact? |
|:------------|:------------------------------------|:----------|:-------|
| `:ns64`     | Network simplex *(default)*         | `Float64` | yes    |
| `:ot64`     | Network simplex with arc mixing     | `Float64` | yes    |
| `:ns32`     | Network simplex                     | `Float32` | yes    |
| `:ot32`     | Network simplex with arc mixing     | `Float32` | yes    |
| `:sinkhorn` | Entropy-regularised OT (Sinkhorn)   | `Float64` | no     |

The compatibility symbols map to the typed strategies `EnergyFlow.NS64`,
`EnergyFlow.OT64`, `EnergyFlow.NS32`, `EnergyFlow.OT32`, and
`EnergyFlow.Sinkhorn`, respectively. Either form may be passed. The canonical
symbol is stored and returned. An unknown symbol raises `ArgumentError`.

# Example
```julia
set_backend(:ot64)
emd(ev0, ev1)              # now uses :ot64
emd(ev0, ev1; backend=:ns64)  # per-call override
emd(ev0, ev1; backend=EnergyFlow.NS64)  # typed per-call override
```

See also [`get_backend`](@ref).
"""
function set_backend(backend::Union{Symbol,EMDBackend})
    strategy = _as_backend(backend)
    symbol = _backend_symbol(strategy)

    EMD_BACKEND[] = symbol
    return symbol
end

"""
    get_backend() -> Symbol

Return the current global default EMD backend (initially `:ns64`).

See also [`set_backend`](@ref).
"""
get_backend() = EMD_BACKEND[]

function _emd!(backend::NetworkSimplexBackend{V},
               ws::EMDWorkspace{V},
               ev0::AbstractMatrix{<:Real},
               ev1::AbstractMatrix{<:Real};
               gdim::Union{Nothing,Int} = nothing,
               n_iter_max::Int = 100_000,
               metric::Union{Nothing,GroundMetric} = nothing,
               strict::Bool = false) where {V}
    selected_metric = something(metric, ws.metric)

    return _emd_backend!(
        backend, ws, ev0, ev1;
        gdim,
        n_iter_max,
        metric=selected_metric,
        strict,
    )
end

function _emd!(::SinkhornBackend,
               ws::SinkhornWorkspace,
               ev0::AbstractMatrix{<:Real},
               ev1::AbstractMatrix{<:Real};
               gdim::Union{Nothing,Int} = nothing,
               metric::Union{Nothing,GroundMetric} = nothing,
               return_flow::Bool = false,
               strict::Bool = false)
    if metric !== nothing && !(metric isa EuclideanMetric)
        throw(ArgumentError(
            "the Sinkhorn backend currently supports only EuclideanMetric()"
        ))
    end

    return emd_sinkhorn!(
        ws, ev0, ev1;
        gdim, return_flow, strict,
    )
end

function _emd!(backend::NetworkSimplexBackend,
               ::SinkhornWorkspace,
               ::AbstractMatrix{<:Real},
               ::AbstractMatrix{<:Real};
               kwargs...)
    symbol = _backend_symbol(backend)
    throw(ArgumentError(
        "backend :$symbol requires an EMDWorkspace"
    ))
end

function _emd!(backend::NetworkSimplexBackend{B},
               ::EMDWorkspace{W},
               ::AbstractMatrix{<:Real},
               ::AbstractMatrix{<:Real};
               kwargs...) where {B,W}
    symbol = _backend_symbol(backend)
    throw(ArgumentError(
        "backend :$symbol uses $B, but the workspace uses $W"
    ))
end


# ─────────────────────────────────────────────────────────────────────
# emd! — workspace-reusing, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emd!(ws, ev0, ev1; backend=get_backend(), gdim=nothing,
         n_iter_max=100_000, metric=nothing, strict=false)

Compute the EMD between two events using a pre-allocated workspace,
avoiding per-call allocations. This is the fastest path when computing many
EMDs in a loop.

The EMD parameters (`beta`, `R`, and `norm`) are read from the workspace.
For an `EMDWorkspace`, `metric=nothing` uses the workspace metric and an
explicit `metric` temporarily overrides it for that call. Sinkhorn currently
supports only `EuclideanMetric`.

# Arguments
- `ws`: an [`EMDWorkspace`](@ref) (for the network-simplex backends) or a
  [`SinkhornWorkspace`](@ref) (for the Sinkhorn backend). Must be large enough
  for the events being compared.
- `ev0`, `ev1`: event matrices, `M × (1 + d)` with column 1 = particle weights
  and columns `2:(1+d)` = coordinates.

# Keywords
- `backend`: compatibility symbol or typed strategy. It must match the
  workspace family, and an exact backend's precision must match the
  `EMDWorkspace` element type. A `SinkhornWorkspace` defaults to `:sinkhorn`.
- `gdim`: number of coordinate dimensions to use (`nothing` = all columns
  after the first).
- `n_iter_max`: maximum network-simplex iterations. It is accepted but ignored
  for a `SinkhornWorkspace`; configure Sinkhorn iterations on the workspace.
- `metric`: optional ground-metric override for an `EMDWorkspace`.
- `return_flow`: supported with a `SinkhornWorkspace`; if `true`, also return
  the transport plan.
- `strict`: if `true`, raise an error when the solver does not converge.

# Returns
The EMD value (`Float64` or `Float32` matching the workspace precision). With
a `SinkhornWorkspace` and `return_flow=true`, returns
`(emd_value, transport_plan)`. The exact `EMDWorkspace` overload returns only
the value; use [`emd`](@ref) when an exact transport plan is needed.

# Example
```julia
n = maximum(size(e, 1) for e in events)
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
vals = [emd!(ws, events[1], e) for e in events[2:end]]
```

See also [`emd`](@ref), [`emds!`](@ref).
"""
function emd!(ws::EMDWorkspace,
              ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
              backend::Union{Symbol,EMDBackend} = EMD_BACKEND[],
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::Union{Nothing,GroundMetric} = nothing,
              strict::Bool = false)
    return _emd!(
        _as_backend(backend), ws, ev0, ev1;
        gdim, n_iter_max, metric, strict,
    )
end

# Sinkhorn workspace variant of emd!
function emd!(ws::SinkhornWorkspace,
              ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
              backend::Union{Symbol,EMDBackend} = :sinkhorn,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::Union{Nothing,GroundMetric} = nothing,
              return_flow::Bool = false,
              strict::Bool = false)
    return _emd!(
        _as_backend(backend), ws, ev0, ev1;
        gdim, metric, return_flow, strict,
    )
end

function _emd!(::SinkhornBackend,
               ::EMDWorkspace,
               ::AbstractMatrix{<:Real},
               ::AbstractMatrix{<:Real};
               kwargs...)
    throw(ArgumentError(
        "the Sinkhorn backend requires a SinkhornWorkspace"
    ))
end

function _emd(backend::NetworkSimplexBackend,
              ev0::AbstractMatrix{<:Real},
              ev1::AbstractMatrix{<:Real};
              R::Real = 1.0,
              beta::Real = 1.0,
              norm::Bool = false,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::GroundMetric = EuclideanMetric(),
              return_flow::Bool = false,
              strict::Bool = false)
    return _emd_backend(
        backend, ev0, ev1;
        R, beta, norm, gdim, n_iter_max, metric, return_flow, strict,
    )
end

function _emd(::SinkhornBackend,
              ev0::AbstractMatrix{<:Real},
              ev1::AbstractMatrix{<:Real};
              R::Real = 1.0,
              beta::Real = 1.0,
              norm::Bool = false,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::GroundMetric = EuclideanMetric(),
              return_flow::Bool = false,
              strict::Bool = false)
    if !(metric isa EuclideanMetric)
        throw(ArgumentError(
            "the Sinkhorn backend currently supports only EuclideanMetric()"
        ))
    end

    return emd_sinkhorn(
        ev0, ev1;
        R, beta, norm, gdim, n_iter_max, return_flow, strict,
    )
end

# ─────────────────────────────────────────────────────────────────────
# emd — allocating convenience, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emd(ev0, ev1; backend=get_backend(), R=1.0, beta=1.0, norm=false,
        gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric(), return_flow=false)

Compute the Energy Mover's Distance between two events. Allocates a fresh
workspace each call; for repeated calls prefer [`emd!`](@ref) with a
pre-allocated workspace.

# Arguments
- `ev0`, `ev1`: event matrices, `M × (1 + d)` with one row per particle.
  Column 1 = particle weights (typically ``p_T``); columns `2:(1+d)` =
  ground-space coordinates (typically rapidity and azimuth).

# Keywords
- `backend`: solver backend, either a compatibility symbol such as `:ns64`
  or a typed backend strategy such as `EnergyFlow.NS64`.
- `R`: distance scale; ground distances are divided by `R` before
  exponentiation. With `norm=false`, `R` controls the relative cost of
  transporting vs. creating/destroying energy (choose `R ≥ half` the maximum
  ground distance for a true metric).
- `beta`: exponent applied to the scaled ground distance (angular exponent).
- `norm`: if `true`, normalize each event's weights to sum to 1 before
    solving. If `false` (default) and the total weights differ, the weight
    difference is handled by a fictitious particle at unit cost, matching the
    Python EnergyFlow convention.
- `gdim`: number of coordinate dimensions to use (`nothing` = all columns
  after the first).
- `n_iter_max`: maximum network-simplex iterations; accepted but ignored by
  the Sinkhorn backend.
- `metric`: ground distance metric, a [`GroundMetric`](@ref) instance
    (default [`EuclideanMetric()`](@ref EuclideanMetric)).
- `return_flow`: if `true`, also return the optimal transport plan matrix.
- `strict`: if `true`, raise an error when the selected solver does not
  converge.

# Returns
The EMD value by default. If `return_flow=true`, returns `(emd_value,
transport_plan)`. When `norm=true`, the transport plan is normalized so its
row/column sums are 1 and the marginals sum to 1. When `norm=false`, the
internal solve is performed on a scaled balanced problem, and the returned plan
is rescaled back to the original physical weights, including any fictitious
source/target row or column introduced by the solver so its marginals match
those original weights.

# Example
```julia
ev0 = [1.0 0.5 0.1;  0.8 -0.3 0.4]   # 2 particles: pT, y, φ
ev1 = [0.9 0.4 0.0;  0.7 -0.2 0.3]

emd(ev0, ev1)                                  # default backend/metric
emd(ev0, ev1; R=0.4, beta=2.0, norm=true)      # EMD parameters
emd(ev0, ev1; backend=:sinkhorn)               # approximate solver
emd(ev0, ev1; backend=EnergyFlow.NS64)          # typed strategy
emd(ev0, ev1; metric=EtaPhiMetric())           # periodic φ handling
```

See also [`emds`](@ref), [`emd!`](@ref).
"""
function emd(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
             backend::Union{Symbol,EMDBackend} = EMD_BACKEND[],
             R::Real         = 1.0,
             beta::Real      = 1.0,
             norm::Bool      = false,
             gdim::Union{Nothing,Int} = nothing,
             n_iter_max::Int = 100_000,
             metric::GroundMetric = EuclideanMetric(),
             return_flow::Bool = false,
             strict::Bool = false)
    return _emd(
        _as_backend(backend), ev0, ev1;
        R, beta, norm, gdim, n_iter_max, metric, return_flow, strict,
    )
end

function _emds!(
    backend::NetworkSimplexBackend{V},
    results::AbstractVector{V},
    events0::AbstractVector{<:AbstractMatrix{<:Real}};
    kwargs...,
) where {V}
    return _emds_backend!(
        backend, results, events0;
        kwargs...,
    )
end

function _emds!(
    backend::NetworkSimplexBackend{B},
    ::AbstractVector{R},
    ::AbstractVector{<:AbstractMatrix{<:Real}};
    kwargs...,
) where {B,R}
    symbol = _backend_symbol(backend)
    throw(ArgumentError(
        "backend :$symbol uses $B, but the results vector uses $R"
    ))
end

function _emds!(
    ::SinkhornBackend,
    ::AbstractVector,
    ::AbstractVector{<:AbstractMatrix{<:Real}};
    kwargs...,
)
    throw(ArgumentError(
        "emds! is not supported for the Sinkhorn backend; use emds instead"
    ))
end

# ─────────────────────────────────────────────────────────────────────
# emds! — in-place pairwise, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emds!(results, events0; backend=get_backend(), R=1.0, beta=1.0, norm=false,
          gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric())

In-place self-pairwise EMD: computes all `n*(n-1)/2` pairwise distances among
`events0` and writes them into the pre-allocated `results` vector (flat
upper-triangular, SciPy `pdist` row-major order — see [`emds`](@ref)).

Computation is multithreaded over pairs. Keywords are as in [`emds`](@ref).
The `results` element type must match the exact backend precision. Sinkhorn is
not supported by `emds!`; use [`emds`](@ref) instead.

# Example
```julia
n = length(events)
results = Vector{Float64}(undef, n * (n - 1) ÷ 2)
emds!(results, events; norm=true)
```
"""
function emds!(results::AbstractVector{<:AbstractFloat},
               events0::AbstractVector{<:AbstractMatrix{<:Real}};
               backend::Union{Symbol,EMDBackend} = EMD_BACKEND[],
               R::Real         = 1.0,
               beta::Real      = 1.0,
               norm::Bool      = false,
               gdim::Union{Nothing,Int} = nothing,
               n_iter_max::Int = 100_000,
               metric::GroundMetric = EuclideanMetric(),
               strict::Bool = false)
    return _emds!(
        _as_backend(backend), results, events0;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end

function _emds(
    backend::NetworkSimplexBackend,
    events0::AbstractVector{<:AbstractMatrix{<:Real}},
    ::Nothing;
    kwargs...,
)
    return _emds_backend(backend, events0; kwargs...)
end

function _emds(
    backend::NetworkSimplexBackend,
    events0::AbstractVector{<:AbstractMatrix{<:Real}},
    events1::AbstractVector{<:AbstractMatrix{<:Real}};
    kwargs...,
)
    return _emds_backend(backend, events0, events1; kwargs...)
end

function _emds(
    ::SinkhornBackend,
    events0::AbstractVector{<:AbstractMatrix{<:Real}},
    events1::Union{Nothing,AbstractVector{<:AbstractMatrix{<:Real}}};
    kwargs...,
)
    return emds_sinkhorn(events0, events1; kwargs...)
end

# ─────────────────────────────────────────────────────────────────────
# emds — allocating pairwise, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emds(events0, events1=nothing; backend=get_backend(), R=1.0, beta=1.0,
         norm=false, gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric())

Compute pairwise Energy Mover's Distances between collections of events.
Computation is multithreaded over pairs (start Julia with `-t auto` or set
`JULIA_NUM_THREADS` to benefit).

# Arguments
- `events0`: vector of event matrices (each `M × (1 + d)`).
- `events1`: `nothing` (default) for self-pairwise mode, or a second vector
  of events for cross-pairwise mode.

# Keywords
Same as [`emd`](@ref): `backend`, `R`, `beta`, `norm`, `gdim`, `n_iter_max`,
`metric`, and `strict`. The `backend` may be a compatibility symbol or a typed
strategy. Sinkhorn supports only `EuclideanMetric` through this interface.

# Returns
- **Self-pairwise** (`events1 === nothing`): a `Vector` of length
  `n*(n-1)/2` containing the strict upper triangle of the distance matrix in
  SciPy `pdist` (row-major) order — i.e. `(1,2), (1,3), …, (1,n), (2,3), …`.
- **Cross-pairwise**: a `Matrix` `D` of size
  `(length(events0), length(events1))` with `D[i, j] = emd(events0[i], events1[j])`.

Element type is `Float64`, or `Float32` for the `:ns32`/`:ot32` backends and
their corresponding typed strategies.

# Example
```julia
events = load_hepmc3_events("events.hepmc"; maxevents=20)

dists = emds(events[1:10]; norm=true)              # 45-element Vector
D     = emds(events[1:10], events[11:20]; norm=true)  # 10×10 Matrix
```

See also [`emd`](@ref), [`emds!`](@ref).
"""
function emds(events0::AbstractVector{<:AbstractMatrix{<:Real}},
              events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
              backend::Union{Symbol,EMDBackend} = EMD_BACKEND[],
              R::Real         = 1.0,
              beta::Real      = 1.0,
              norm::Bool      = false,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::GroundMetric = EuclideanMetric(),
              strict::Bool = false)
    return _emds(
        _as_backend(backend), events0, events1;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end
