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
    set_backend(backend::Symbol)

Set the default EMD backend. Available: `:ns64`, `:ot64`, `:ns32`, `:ot32`, `:sinkhorn`.
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

Return the current default EMD backend.
"""
get_backend() = EMD_BACKEND[]

# ─────────────────────────────────────────────────────────────────────
# emd! — workspace-reusing, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emd!(ws, ev0, ev1; backend=get_backend(), gdim=nothing, n_iter_max=100_000) -> Float64

Compute EMD between two events using pre-allocated workspace `ws`.
Dispatches to the active backend.

# Arguments
- `ws::EMDWorkspace`: pre-allocated workspace (beta, R, norm are read from ws).
- `ev0`, `ev1`: EnergyFlow-format event matrices (M×(1+gdim)).
- `backend`: backend to use (default: current global backend).
- `gdim`: coordinate dimensions to use (`nothing` = all).
- `n_iter_max`: max iterations for iterative solvers.

# Returns
- `Float64` EMD value.
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
              metric::Union{Nothing,GroundMetric} = nothing)
    return emd_sinkhorn!(ws, ev0, ev1; gdim=gdim)
end

# ─────────────────────────────────────────────────────────────────────
# emd — allocating convenience, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emd(ev0, ev1; backend=get_backend(), R=1.0, beta=1.0, norm=false,
        gdim=nothing, n_iter_max=100_000) -> Float64

Compute EMD between two events. Allocates a fresh workspace.
For repeated calls, prefer `emd!` with a pre-allocated workspace.

# Arguments
- `ev0`, `ev1`: M×(1+gdim) event matrices.
- `backend`: backend to use (default: current global backend).
- `R`, `beta`, `norm`: EMD parameters.
- `gdim`, `n_iter_max`: see `emd!`.

# Returns
- `Float64` EMD value.
"""
function emd(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
             backend::Symbol = EMD_BACKEND[],
             R::Real         = 1.0,
             beta::Real      = 1.0,
             norm::Bool      = false,
             gdim::Union{Nothing,Int} = nothing,
             n_iter_max::Int = 100_000,
             metric::GroundMetric = EuclideanMetric())

    if backend === :ns64
        return emd_ns64(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot64
        return emd_ot64(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ns32
        return emd_ns32(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :ot32
        return emd_ot32(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max, metric=metric)
    elseif backend === :sinkhorn
        return emd_sinkhorn(ev0, ev1; R=R, beta=beta, norm=norm, gdim=gdim, n_iter_max=n_iter_max)
    else
        error("Unknown backend :$backend. Available: $(AVAILABLE_BACKENDS)")
    end
end

# ─────────────────────────────────────────────────────────────────────
# emds! — in-place pairwise, backend-dispatching
# ─────────────────────────────────────────────────────────────────────

"""
    emds!(results, events0; backend=get_backend(), R=1.0, beta=1.0, norm=false,
          gdim=nothing, n_iter_max=100_000)

In-place self-pairwise EMD: writes to pre-allocated `results` vector
(flat upper-triangular, length n*(n-1)/2). Dispatches to the active backend.
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
         norm=false, gdim=nothing, n_iter_max=100_000)

Compute pairwise EMDs. Dispatches to the active backend.

# Returns
- Self-pairwise (`events1=nothing`): `Vector{Float64}` of length `n*(n-1)/2`.
- Cross-pairwise: `Matrix{Float64}` of shape `(length(events0), length(events1))`.
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
