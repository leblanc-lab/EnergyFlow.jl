#=
    EMDSolver.jl

    EMDSolver: Network Simplex backend for single-pair EMD computation.

    Provides EMDWorkspace for pre-allocated computation, _emd_raw! for the
    core solver loop, and Float64/Float32 single-pair frontends
    (emd_ns64/ot64, emd_ns32/ot32 and their in-place variants).

    The EMD is defined as:
        EMD_{β,R}(P, Q) = min_{f≥0} Σ_ij f_ij (d_ij/R)^β
    subject to transport constraints.

    When norm=false and total weights differ, a fictitious particle is added
    to the lighter distribution with distance 1 to all particles in the other.
=#

# ─────────────────────────────────────────────────────────────────────
# EMDWorkspace — pre-allocated workspace for repeated EMD computations
# ─────────────────────────────────────────────────────────────────────

"""
    EMDWorkspace{V<:AbstractFloat, M<:GroundMetric}

Pre-allocated workspace for repeated EMD computations with the
network-simplex backends. Construct once, then pass to [`emd!`](@ref) (or the
backend-specific `emd_ns64!` etc.) to avoid per-call allocations.

The EMD parameters (`beta`, `R`, `norm`, `metric`) are stored in the
workspace and used by every solve.

!!! note
    The workspace is specialized on its ground-metric type `M` (the second
    type parameter). Updating `beta`, `R`, or `norm` in place is fine, but
    switching to a *different* metric type requires constructing a new
    workspace — assigning a metric of another type to `ws.metric` will error,
    because the field is typed `M`.

# Fields
- `ns`: [`NetworkSimplexSolver`](@ref) workspace
- `beta`, `R`: EMD parameters
- `norm`: if `true`, normalize event weights to sum to 1 before solving
- `metric`: ground distance metric ([`GroundMetric`](@ref)); its concrete type
  is the workspace's `M` parameter
- `parallel_threshold`: fill the cost matrix in parallel when
  `n0 * n1 >= parallel_threshold` (default `40_000`)

See also [`SinkhornWorkspace`](@ref).
"""
mutable struct EMDWorkspace{V<:AbstractFloat, M<:GroundMetric}
    ns::NetworkSimplexSolver{V}
    beta::V
    R::V
    norm::Bool
    max_n0::Int
    max_n1::Int

    # Temporary buffers for weights (with possible fictitious particle)
    source_weights::Vector{V}
    target_weights::Vector{V}

    # Parallel cost fill threshold
    parallel_threshold::Int

    # Ground distance metric
    metric::M
end

"""
    EMDWorkspace(max_n0, max_n1; beta=1.0, R=1.0, norm=true, metric=EuclideanMetric())
    EMDWorkspace{V}(max_n0, max_n1; kwargs...)

Create a workspace for computing EMDs between events of up to `max_n0` and
`max_n1` particles. The unparameterized form defaults to `Float64`; use
`EMDWorkspace{Float32}` for the `:ns32`/`:ot32` backends.

# Arguments
- `max_n0`, `max_n1`: maximum particle counts of the two events in any
  subsequent `emd!` call (one extra slot is reserved internally for the
  fictitious particle used when `norm=false`).

# Keywords
- `beta`, `R`, `norm`, `metric`: EMD parameters applied to every solve with
  this workspace; see [`emd`](@ref) for their meaning. Note that `norm`
  defaults to `true` here, unlike `emd`.

# Example
```julia
n = maximum(size(e, 1) for e in events)
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
val = emd!(ws, events[1], events[2])
```
"""
function EMDWorkspace{V}(max_n0::Int, max_n1::Int;
                         beta::Real = 1.0, R::Real = 1.0, norm::Bool = true,
                         metric::GroundMetric = EuclideanMetric()) where V
    # +1 for possible fictitious particle
    ns = NetworkSimplexSolver{V}(max_n0 + 1, max_n1 + 1)
    EMDWorkspace{V, typeof(metric)}(
        ns, V(beta), V(R), norm,
        max_n0, max_n1,
        Vector{V}(undef, max_n0 + 1),
        Vector{V}(undef, max_n1 + 1),
        40_000, # parallel_threshold: parallel cost fill when n0*n1 >= this
        metric,
    )
end

EMDWorkspace(max_n0::Int, max_n1::Int; kwargs...) = EMDWorkspace{Float64}(max_n0, max_n1; kwargs...)

"""
    _handle_solver_status(status; strict=false, backend=:ns64, context="EMD solve")

Internal helper: validates solver status. For non-`:optimal` status, emits a
warning by default or throws when `strict=true`.
"""
function _handle_solver_status(status::Symbol;
                               strict::Bool = false,
                               backend::Symbol = :ns64,
                               context::AbstractString = "EMD solve",
                               value::Union{Real, Nothing} = nothing)
    status === :optimal && return

    if backend === :sinkhorn && status === :max_iter && value !== nothing && isfinite(value)
        if strict
            error("$context failed with backend :$backend (status=:$status). Returned value may be invalid.")
        end
        return
    end

    msg = "$context failed with backend :$backend (status=:$status). Returned value may be invalid."
    if strict
        error(msg)
    else
        @warn msg
    end
end

"""
    _handle_pairwise_statuses(statuses; strict=false, backend=:ns64,
                             context="pairwise EMD solve")

Aggregate per-pair solver statuses and emit a single summary warning/error.
This avoids warning spam in threaded pairwise computations while preserving the
existing strict mode behavior.
"""
function _handle_pairwise_statuses(statuses::AbstractVector{Symbol};
                                 strict::Bool = false,
                                 backend::Symbol = :ns64,
                                 context::AbstractString = "pairwise EMD solve")
    isempty(statuses) && return
    bad = count(!isequal(:optimal), statuses)
    bad == 0 && return

    summary = "$context had $bad out of $(length(statuses)) non-optimal results with backend :$backend."
    if strict
        error(summary * " Statuses: $(join(statuses, ", ")).")
    else
        @warn summary * " Returned values may be invalid."
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
# Raw EMD computation (weights + coords interface)
# ─────────────────────────────────────────────────────────────────────

"""
    _emd_raw!(ws, weights0, coords0, weights1, coords1; max_iter=100_000,
              metric=ws.metric) -> (V, Symbol)

Internal: compute EMD between two distributions using pre-allocated workspace.
Takes raw weights and coordinates. Returns (emd_value, status).
"""
function _emd_raw!(ws::EMDWorkspace{V},
                   weights0::AbstractVector{V}, coords0::AbstractMatrix{V},
                   weights1::AbstractVector{V}, coords1::AbstractMatrix{V};
                   max_iter::Int = 100_000,
                   arc_mixing::Bool = false,
                   metric::GroundMetric = ws.metric) where V

    n0 = length(weights0)
    n1 = length(weights1)

    if n0 == 0 && n1 == 0
        return zero(V), :optimal
    end

    total0 = sum(weights0)
    total1 = sum(weights1)

    if ws.norm && (total0 == zero(V) || total1 == zero(V))
        return zero(V), :optimal
    end

    # Determine if we need a fictitious particle
    has_fict_source = false
    has_fict_target = false
    scale = one(V)

    if ws.norm
        # Normalize weights to sum to 1
        n0_eff = n0
        n1_eff = n1
        @inbounds for i in 1:n0
            ws.source_weights[i] = weights0[i] / total0
        end
        @inbounds for j in 1:n1
            ws.target_weights[j] = weights1[j] / total1
        end
    else
        weight_diff = total1 - total0

        if weight_diff > zero(V)
            # Target is heavier: add fictitious source
            has_fict_source = true
            n0_eff = n0 + 1
            n1_eff = n1
        elseif weight_diff < zero(V)
            # Source is heavier: add fictitious target
            has_fict_target = true
            n0_eff = n0
            n1_eff = n1 + 1
        else
            n0_eff = n0
            n1_eff = n1
        end

        # Scale weights for numerical stability
        max_total = max(total0, total1)
        scale = max_total

        @inbounds for i in 1:n0
            ws.source_weights[i] = weights0[i] / max_total
        end
        @inbounds for j in 1:n1
            ws.target_weights[j] = weights1[j] / max_total
        end

        # Add fictitious particle weight
        if has_fict_source
            ws.source_weights[n0_eff] = abs(weight_diff) / max_total
        end
        if has_fict_target
            ws.target_weights[n1_eff] = abs(weight_diff) / max_total
        end
    end

    # Fill cost matrix — parallel or serial based on threshold
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff
    if n0_real * n1_real >= ws.parallel_threshold
        _fill_costs_parallel!(ws, metric, coords0, coords1, n0_eff, n1_eff, has_fict_source, has_fict_target)
    else
        _fill_costs!(ws, metric, coords0, coords1, n0_eff, n1_eff, has_fict_source, has_fict_target)
    end

    # Solve
    old_arc_mixing = ws.ns.arc_mixing
    ws.ns.arc_mixing = arc_mixing
    sw = @view ws.source_weights[1:n0_eff]
    tw = @view ws.target_weights[1:n1_eff]
    status = network_simplex!(ws.ns, sw, tw; max_iter=max_iter)
    ws.ns.arc_mixing = old_arc_mixing

    # Avoid returning stale ws.ns.total_cost when the solver fails (e.g. :max_iter).
    if status !== :optimal
        return V(NaN), status
    end

    # Extract total cost only on successful solve.
    emd_val = ws.ns.total_cost
    if !ws.norm
        emd_val *= scale
    end

    return emd_val, status
end

# Shared implementation for the workspace-reusing single-pair frontends.
function _emd_backend!(::Type{U}, ws::EMDWorkspace{V},
                       ev0::AbstractMatrix{<:Real},
                       ev1::AbstractMatrix{<:Real};
                       arc_mixing::Bool,
                       backend::Symbol,
                       context::AbstractString,
                       gdim::Union{Nothing,Int} = nothing,
                       n_iter_max::Int = 100_000,
                       metric::GroundMetric = ws.metric,
                       strict::Bool = false) where {U<:AbstractFloat,V}

    w0, c0 = _unpack_event(U, ev0, gdim)
    w1, c1 = _unpack_event(U, ev1, gdim)

    val, status = _emd_raw!(
        ws,
        convert(Vector{V}, w0), convert(Matrix{V}, c0),
        convert(Vector{V}, w1), convert(Matrix{V}, c1);
        max_iter=n_iter_max,
        arc_mixing,
        metric,
    )

    _handle_solver_status(status; strict, backend, context)
    return val
end

# Shared implementation for the allocating single-pair frontends.
function _emd_backend(::Type{V},
                      ev0::AbstractMatrix{<:Real},
                      ev1::AbstractMatrix{<:Real};
                      arc_mixing::Bool,
                      backend::Symbol,
                      context::AbstractString,
                      R::Real = 1.0,
                      beta::Real = 1.0,
                      norm::Bool = false,
                      gdim::Union{Nothing,Int} = nothing,
                      n_iter_max::Int = 100_000,
                      metric::GroundMetric = EuclideanMetric(),
                      return_flow::Bool = false,
                      strict::Bool = false) where {V<:AbstractFloat}

    w0, c0 = _unpack_event(V, ev0, gdim)
    w1, c1 = _unpack_event(V, ev1, gdim)

    ws = EMDWorkspace{V}(
        length(w0), length(w1);
        beta,
        R,
        norm,
        metric,
    )

    val, status = _emd_raw!(
        ws, w0, c0, w1, c1;
        max_iter=n_iter_max,
        arc_mixing,
    )

    _handle_solver_status(status; strict, backend, context)

    return return_flow ?
           (val, _transport_plan(ws.ns; arc_mixing)) :
           val
end

# ─────────────────────────────────────────────────────────────────────
# emd_ns64! / emd_ns64 — single-pair NS Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emd_ns64!(ws, ev0, ev1; gdim=nothing, n_iter_max=100_000) -> Float64

Compute EMD using the Network Simplex Float64 backend with a pre-allocated
`EMDWorkspace`. The workspace's `beta`, `R`, `norm` settings are used.

# Arguments
- `ws::EMDWorkspace`: pre-allocated workspace.
- `ev0`, `ev1`: EnergyFlow-format event matrices (M×(1+gdim)).
  Column 1 = particle weights (pT); columns 2:(1+gdim) = coordinates.
- `gdim`: coordinate dimensions to use (`nothing` = all).
- `n_iter_max`: max network-simplex iterations.

# Returns
- `Float64` EMD value.
"""
function emd_ns64!(ws::EMDWorkspace{V},
                   ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = ws.metric,
                   strict::Bool = false) where V
    return _emd_backend!(Float64, ws, ev0, ev1;
            arc_mixing=false, backend=:ns64, context="emd_ns64!",
            gdim, n_iter_max, metric, strict)
end

"""
    emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=false, gdim=nothing,
             n_iter_max=100_000, metric=EuclideanMetric()) -> Float64

Compute EMD using the Network Simplex Float64 backend.
Allocates a fresh workspace each call. For repeated calls, prefer `emd_ns64!`.

Equivalent to [`emd`](@ref) with `backend=:ns64`.

# Arguments
- `ev0`, `ev1`: M×(1+gdim) matrices. Column 1 = particle weights (pT);
  columns 2:(1+gdim) = spatial coordinates.
- `R`: distance scale parameter (default 1.0).
- `beta`: distance exponent (default 1.0).
- `norm`: normalize weights to sum to 1 before solving (default false).
- `gdim`: number of coordinate dimensions to use. `nothing` = use all remaining columns.
- `n_iter_max`: max network-simplex iterations (default 100_000).
- `metric`: ground distance metric (default [`EuclideanMetric()`](@ref EuclideanMetric)).

# Returns
- `Float64` EMD value.
"""
function emd_ns64(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                  R::Real        = 1.0,
                  beta::Real     = 1.0,
                  norm::Bool     = false,
                  gdim::Union{Nothing,Int} = nothing,
                  n_iter_max::Int = 100_000,
                  metric::GroundMetric = EuclideanMetric(),
                  return_flow::Bool = false,
                  strict::Bool = false)
    return _emd_backend(Float64, ev0, ev1;
                    arc_mixing=false, backend=:ns64, context="emd_ns64",
                    R, beta, norm, gdim, n_iter_max, metric, return_flow,
                    strict)
end

# ─────────────────────────────────────────────────────────────────────
# emd_ot64! / emd_ot64 — OT-style (arc mixing) NS Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emd_ot64!(ws, ev0, ev1; gdim=nothing, n_iter_max=100_000) -> Float64

Compute EMD using the OT-style Float64 backend (arc mixing enabled).
Uses the same NetworkSimplex solver but with POT-style arc interleaving
for improved performance on unbalanced transport problems.
"""
function emd_ot64!(ws::EMDWorkspace{V},
                   ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = ws.metric,
                   strict::Bool = false) where V
    return _emd_backend!(Float64, ws, ev0, ev1;
        arc_mixing=true, backend=:ot64, context="emd_ot64!",
        gdim, n_iter_max, metric, strict)
end

"""
    emd_ot64(ev0, ev1; R=1.0, beta=1.0, norm=false, gdim=nothing,
             n_iter_max=100_000, metric=EuclideanMetric()) -> Float64

Compute EMD using the OT-style Float64 backend (arc mixing enabled).
Allocates a fresh workspace each call. For repeated calls, prefer `emd_ot64!`.
Arguments are as in [`emd_ns64`](@ref).
"""
function emd_ot64(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                  R::Real        = 1.0,
                  beta::Real     = 1.0,
                  norm::Bool     = false,
                  gdim::Union{Nothing,Int} = nothing,
                  n_iter_max::Int = 100_000,
                  metric::GroundMetric = EuclideanMetric(),
                  return_flow::Bool = false,
                  strict::Bool = false)
    return _emd_backend(Float64, ev0, ev1;
                        arc_mixing=true, backend=:ot64, context="emd_ot64",
                        R, beta, norm, gdim, n_iter_max, metric, return_flow,
                        strict)
end

# ═════════════════════════════════════════════════════════════════════
# Float32 backends (ns32 / ot32)
# ═════════════════════════════════════════════════════════════════════

# ─── emd_ns32! / emd_ns32 — single-pair NS Float32 backend ─────────

"""
    emd_ns32!(ws, ev0, ev1; gdim=nothing, n_iter_max=100_000) -> Float32

Compute EMD using the Network Simplex Float32 backend with a pre-allocated
`EMDWorkspace{Float32}`.
"""
function emd_ns32!(ws::EMDWorkspace{Float32},
                   ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = ws.metric,
                   strict::Bool = false)
    return _emd_backend!(Float32, ws, ev0, ev1;
            arc_mixing=false, backend=:ns32, context="emd_ns32!",
            gdim, n_iter_max, metric, strict)
end

"""
    emd_ns32(ev0, ev1; R=1.0, beta=1.0, norm=false, gdim=nothing,
             n_iter_max=100_000, metric=EuclideanMetric()) -> Float32

Compute EMD using the Network Simplex Float32 backend.
Allocates a fresh workspace each call. For repeated calls, prefer `emd_ns32!`.
Arguments are as in [`emd_ns64`](@ref).
"""
function emd_ns32(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                  R::Real        = 1.0,
                  beta::Real     = 1.0,
                  norm::Bool     = false,
                  gdim::Union{Nothing,Int} = nothing,
                  n_iter_max::Int = 100_000,
                  metric::GroundMetric = EuclideanMetric(),
                  return_flow::Bool = false,
                  strict::Bool = false)
    return _emd_backend(Float32, ev0, ev1;
                        arc_mixing=false, backend=:ns32, context="emd_ns32",
                        R, beta, norm, gdim, n_iter_max, metric, return_flow,
                        strict)
end

# ─── emd_ot32! / emd_ot32 — OT-style (arc mixing) Float32 backend ──

"""
    emd_ot32!(ws, ev0, ev1; gdim=nothing, n_iter_max=100_000) -> Float32

Compute EMD using the OT-style Float32 backend (arc mixing enabled).
"""
function emd_ot32!(ws::EMDWorkspace{Float32},
                   ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = ws.metric,
                   strict::Bool = false)
    return _emd_backend!(Float32, ws, ev0, ev1;
        arc_mixing=true, backend=:ot32, context="emd_ot32!",
        gdim, n_iter_max, metric, strict)
end

"""
    emd_ot32(ev0, ev1; R=1.0, beta=1.0, norm=false, gdim=nothing,
             n_iter_max=100_000, metric=EuclideanMetric()) -> Float32

Compute EMD using the OT-style Float32 backend (arc mixing enabled).
Allocates a fresh workspace each call. For repeated calls, prefer `emd_ot32!`.
Arguments are as in [`emd_ns64`](@ref).
"""
function emd_ot32(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                  R::Real        = 1.0,
                  beta::Real     = 1.0,
                  norm::Bool     = false,
                  gdim::Union{Nothing,Int} = nothing,
                  n_iter_max::Int = 100_000,
                  metric::GroundMetric = EuclideanMetric(),
                  return_flow::Bool = false,
                  strict::Bool = false)
    return _emd_backend(Float32, ev0, ev1;
                        arc_mixing=true, backend=:ot32, context="emd_ot32",
                        R, beta, norm, gdim, n_iter_max, metric, return_flow,
                        strict)
end
