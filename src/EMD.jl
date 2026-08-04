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

const AVAILABLE_BACKENDS = [:ns64, :ot64, :ns32, :ot32, :sinkhorn]
const EMD_BACKEND = Ref(:ns64)

"""
    set_backend(backend::Symbol) -> Symbol

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

Returns the newly set backend. Throws an error for unknown backends.

# Example
```julia
set_backend(:ot64)
emd(ev0, ev1)              # now uses :ot64
emd(ev0, ev1; backend=:ns64)  # per-call override
```

See also [`get_backend`](@ref).
"""
function set_backend(backend::Symbol)
    if backend ∉ AVAILABLE_BACKENDS
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
    EMD_BACKEND[] = backend
    return backend
end

"""
    get_backend() -> Symbol

Return the current global default EMD backend (initially `:ns64`).

See also [`set_backend`](@ref).
"""
get_backend() = EMD_BACKEND[]

# ─────────────────────────────────────────────────────────────────────
# emd! — workspace-reusing, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emd!(ws, ev0, ev1; backend=get_backend(), gdim=nothing, n_iter_max=100_000, return_flow=false)

Compute the EMD between two events using a pre-allocated workspace,
avoiding per-call allocations. This is the fastest path when computing many
EMDs in a loop.

The EMD parameters (`beta`, `R`, `norm`, `metric`) are read from the
workspace, not passed as keywords — set them when constructing it.

# Arguments
- `ws`: an [`EMDWorkspace`](@ref) (for the network-simplex backends) or a
  [`SinkhornWorkspace`](@ref) (for the Sinkhorn backend). Must be large enough
  for the events being compared.
- `ev0`, `ev1`: event matrices, `M × (1 + d)` with column 1 = particle weights
  and columns `2:(1+d)` = coordinates.

# Keywords
- `backend`: solver backend (default: current global backend; ignored when
  `ws` is a `SinkhornWorkspace`, which always uses `:sinkhorn`).
- `gdim`: number of coordinate dimensions to use (`nothing` = all columns
  after the first).
- `n_iter_max`: maximum solver iterations.

# Returns
The EMD value (`Float64` or `Float32` matching the workspace precision).
If `return_flow=true`, returns `(emd_value, transport_plan)`.

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
              backend::Symbol = EMD_BACKEND[],
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::Union{Nothing,GroundMetric} = nothing)

    metric = something(metric, ws.metric)

    if backend === :ns64
        return emd_ns64!(ws, ev0, ev1; gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot64
        return emd_ot64!(ws, ev0, ev1; gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ns32
        return emd_ns32!(ws, ev0, ev1; gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot32
        return emd_ot32!(ws, ev0, ev1; gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    else
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
end

# Sinkhorn workspace variant of emd!
function emd!(ws::SinkhornWorkspace,
              ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
              backend::Symbol = :sinkhorn,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::Union{Nothing,GroundMetric} = nothing,
              return_flow::Bool = false)
    if metric !== nothing && !(metric isa EuclideanMetric)
        error("sinkhorn backend currently supports only EuclideanMetric().")
    end
    return emd_sinkhorn!(ws, ev0, ev1; gdim=gdim, return_flow=return_flow)
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
- `backend`: solver backend, one of `:ns64` (default), `:ot64`, `:ns32`,
  `:ot32`, `:sinkhorn`. See [`set_backend`](@ref).
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
- `n_iter_max`: maximum solver iterations.
- `metric`: ground distance metric, a [`GroundMetric`](@ref) instance
    (default [`EuclideanMetric()`](@ref EuclideanMetric)).
- `return_flow`: if `true`, also return the optimal transport plan matrix.

# Returns
The EMD value by default. If `return_flow=true`, returns `(emd_value,
transport_plan)`. For `norm=false`, the returned plan includes any fictitious
source/target row or column introduced by the solver so its marginals match
the balanced problem that was actually solved.

# Example
```julia
ev0 = [1.0 0.5 0.1;  0.8 -0.3 0.4]   # 2 particles: pT, y, φ
ev1 = [0.9 0.4 0.0;  0.7 -0.2 0.3]

emd(ev0, ev1)                                  # default backend/metric
emd(ev0, ev1; R=0.4, beta=2.0, norm=true)      # EMD parameters
emd(ev0, ev1; backend=:sinkhorn)               # approximate solver
emd(ev0, ev1; metric=EtaPhiMetric())           # periodic φ handling
```

See also [`emds`](@ref), [`emd!`](@ref).
"""
function emd(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
             backend::Symbol = EMD_BACKEND[],
             R::Real         = 1.0,
             beta::Real      = 1.0,
             norm::Bool      = false,
             gdim::Union{Nothing,Int} = nothing,
             n_iter_max::Int = 100_000,
             metric::GroundMetric = EuclideanMetric(),
             return_flow::Bool = false)

    if backend === :ns64
        return emd_ns64(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric, return_flow=return_flow)
    elseif backend === :ot64
        return emd_ot64(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric, return_flow=return_flow)
    elseif backend === :ns32
        return emd_ns32(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric, return_flow=return_flow)
    elseif backend === :ot32
        return emd_ot32(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric, return_flow=return_flow)
    elseif backend === :sinkhorn
        if metric isa GroundMetric && !(metric isa EuclideanMetric)
            error("sinkhorn backend currently supports only EuclideanMetric().")
        end
        return emd_sinkhorn(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, return_flow=return_flow)
    else
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
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
Not supported for the `:sinkhorn` backend (use `emds` instead).

# Example
```julia
n = length(events)
results = Vector{Float64}(undef, n * (n - 1) ÷ 2)
emds!(results, events; norm=true)
```
"""
function emds!(results::AbstractVector{<:AbstractFloat},
               events0::AbstractVector{<:AbstractMatrix{<:Real}};
               backend::Symbol = EMD_BACKEND[],
               R::Real         = 1.0,
               beta::Real      = 1.0,
               norm::Bool      = false,
               gdim::Union{Nothing,Int} = nothing,
               n_iter_max::Int = 100_000,
               metric::GroundMetric = EuclideanMetric())

    if backend === :ns64
        return emds_ns64!(results, events0; R=R, beta=beta, norm=norm,
                          gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot64
        return emds_ot64!(results, events0; R=R, beta=beta, norm=norm,
                          gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ns32
        return emds_ns32!(results, events0; R=R, beta=beta, norm=norm,
                          gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot32
        return emds_ot32!(results, events0; R=R, beta=beta, norm=norm,
                          gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :sinkhorn
        error("emds! (in-place) is not supported for :sinkhorn backend. Use emds() instead.")
    else
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
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
`metric`.

# Returns
- **Self-pairwise** (`events1 === nothing`): a `Vector` of length
  `n*(n-1)/2` containing the strict upper triangle of the distance matrix in
  SciPy `pdist` (row-major) order — i.e. `(1,2), (1,3), …, (1,n), (2,3), …`.
- **Cross-pairwise**: a `Matrix` `D` of size
  `(length(events0), length(events1))` with `D[i, j] = emd(events0[i], events1[j])`.

Element type is `Float64`, or `Float32` for the `:ns32`/`:ot32` backends.

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
              backend::Symbol = EMD_BACKEND[],
              R::Real         = 1.0,
              beta::Real      = 1.0,
              norm::Bool      = false,
              gdim::Union{Nothing,Int} = nothing,
              n_iter_max::Int = 100_000,
              metric::GroundMetric = EuclideanMetric())

    if backend === :ns64
        return emds_ns64(events0, events1; R=R, beta=beta, norm=norm,
                         gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot64
        return emds_ot64(events0, events1; R=R, beta=beta, norm=norm,
                         gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ns32
        return emds_ns32(events0, events1; R=R, beta=beta, norm=norm,
                         gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot32
        return emds_ot32(events0, events1; R=R, beta=beta, norm=norm,
                         gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :sinkhorn
        return emds_sinkhorn(events0, events1; R=R, beta=beta, norm=norm,
                             gdim=gdim, n_iter_max=n_iter_max)
    else
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
end
