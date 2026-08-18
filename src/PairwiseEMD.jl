#=
    PairwiseEMD.jl

    PairwiseEMD: Pairwise EMD computation.

    Provides self-pairwise and cross-pairwise EMD computation for
    collections of events, using thread-parallel work distribution.

    Includes Float64 and Float32 backends for both NS and OT variants,
    plus in-place versions that write to pre-allocated result vectors.
=#

# ─────────────────────────────────────────────────────────────────────
# Flat index to pair mapping
# ─────────────────────────────────────────────────────────────────────

"""
Map flat upper-triangular index k (1-based) to (i, j) pair with i < j.
Uses the SciPy pdist convention.
"""
function _flat_to_pair(k::Int, n::Int)
    kk = k - 1  # 0-based
    i = Int(floor(n - 0.5 - sqrt((n - 0.5)^2 - 2.0 * kk))) + 1
    j = kk - (i - 1) * (2 * n - i) ÷ 2 + i + 1
    return i, j
end

"""
Run `work!(ws, k)` for every `k in 1:npairs`, distributing the range across
threads block-cyclically — one spawned task and one reusable workspace
(`make_ws()`) per task, with blocks of consecutive pairs dealt round-robin
to the tasks.

The assignment is block-cyclic rather than contiguous because per-pair cost
varies with event multiplicity and correlates with pair index (pairs sharing a
high-multiplicity event `i` are adjacent), so contiguous chunks have unequal
total cost and the slowest chunk sets the wall time. Dealing blocks round-robin
decorrelates cost from task.
"""
function _pairwise_parallel!(npairs::Int, make_ws::F1, work!::F2) where {F1, F2}
    npairs == 0 && return nothing
    ntasks = clamp(Threads.nthreads(), 1, npairs)
    bsize  = clamp(npairs ÷ (32 * ntasks), 1, 64)
    stride = ntasks * bsize
    @sync for t in 1:ntasks
        Threads.@spawn begin
            ws = make_ws()
            for start in ((t - 1) * bsize + 1):stride:npairs
                stop = min(start + bsize - 1, npairs)
                for k in start:stop
                    work!(ws, k)
                end
            end
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
# Pairwise EMD internals (tuple-based interface)
# ─────────────────────────────────────────────────────────────────────

"""
Self-pairwise EMD: upper triangular of n x n (n*(n-1)/2 computations).
"""
function _pairwise_emd_self(
                            backend::NetworkSimplexBackend{V},
                            events::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractMatrix}};
                            beta::Real = 1.0,
                            R::Real = 1.0,
                            norm::Bool = true,
                            max_iter::Int = 100_000,
                            symmetric::Bool = true,
                            metric::GroundMetric = EuclideanMetric(),
                            strict::Bool = false,
                        ) where {V}
    arc_mixing = _arc_mixing(backend)
    nev = length(events)
    if nev <= 1
        return Vector{V}(undef, 0)
    end

    max_n = maximum(length(e[1]) for e in events)
    npairs = nev * (nev - 1) ÷ 2

    results = Vector{V}(undef, npairs)
    statuses = Vector{Symbol}(undef, npairs)

    make_ws() = begin
        w = EMDWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm, metric=metric)
        w.parallel_threshold = typemax(Int)
        w
    end
    work!(ws, k) = begin
        i, j = _flat_to_pair(k, nev)

        w0, c0 = events[i]
        w1, c1 = events[j]

        val, status = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        statuses[k] = status
        results[k] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)
    _handle_pairwise_statuses(
        statuses;
        strict,
        backend=_backend_symbol(backend),
        context="pairwise self! solve",
    )
    if symmetric
        return results
    else
        D = zeros(V, nev, nev)
        k = 1
        for i in 1:nev
            for j in (i+1):nev
                D[i, j] = results[k]
                D[j, i] = results[k]
                k += 1
            end
        end
        return D
    end
end

"""
Cross-pairwise EMD: full m x n matrix.
"""
function _pairwise_emd_cross(
                            backend::NetworkSimplexBackend{V},
                            events_a::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractMatrix}},
                            events_b::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractMatrix}};
                            beta::Real = 1.0,
                            R::Real = 1.0,
                            norm::Bool = true,
                            max_iter::Int = 100_000,
                            metric::GroundMetric = EuclideanMetric(),
                            strict::Bool = false,
                        ) where {V}
    arc_mixing = _arc_mixing(backend)
    na = length(events_a)
    nb = length(events_b)
    if na == 0 || nb == 0
        return Matrix{V}(undef, na, nb)
    end

    max_na = maximum(length(e[1]) for e in events_a)
    max_nb = maximum(length(e[1]) for e in events_b)
    max_n = max(max_na, max_nb)

    D = Matrix{V}(undef, na, nb)
    statuses = Vector{Symbol}(undef, na * nb)

    npairs = na * nb
    make_ws() = begin
        w = EMDWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm, metric=metric)
        w.parallel_threshold = typemax(Int)
        w
    end
    work!(ws, k) = begin
        i = (k - 1) ÷ nb + 1
        j = mod1(k, nb)

        w0, c0 = events_a[i]
        w1, c1 = events_b[j]

        val, status = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        statuses[k] = status
        D[i, j] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)
    _handle_pairwise_statuses(
        statuses;
        strict,
        backend=_backend_symbol(backend),
        context="pairwise cross solve",
    )

    return D
end

"""
In-place self-pairwise: writes to pre-allocated results vector.
"""
function _pairwise_emd_self!(
                            backend::NetworkSimplexBackend{V},
                            results::AbstractVector{V},
                            events::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractMatrix}};
                            beta::Real = 1.0,
                            R::Real = 1.0,
                            norm::Bool = true,
                            max_iter::Int = 100_000,
                            metric::GroundMetric = EuclideanMetric(),
                            strict::Bool = false,
                        ) where {V}
    arc_mixing = _arc_mixing(backend)
    nev = length(events)
    npairs = nev * (nev - 1) ÷ 2
    @assert length(results) >= npairs "results vector too short"
    if nev <= 1
        return results
    end

    statuses = Vector{Symbol}(undef, npairs)
    max_n = maximum(length(e[1]) for e in events)

    make_ws() = begin
        w = EMDWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm, metric=metric)
        w.parallel_threshold = typemax(Int)
        w
    end
    work!(ws, k) = begin
        i, j = _flat_to_pair(k, nev)

        w0, c0 = events[i]
        w1, c1 = events[j]

        val, status = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        statuses[k] = status
        results[k] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)
    _handle_pairwise_statuses(
        statuses;
        strict,
        backend=_backend_symbol(backend),
        context="pairwise self! solve",
    )
    return results
end

function _unpack_events(
    ::NetworkSimplexBackend{V},
    events::AbstractVector{<:AbstractMatrix{<:Real}},
    gdim::Union{Nothing,Int},
) where {V}
    return [_unpack_event(V, event, gdim) for event in events]
end

function _emds_backend(
    backend::NetworkSimplexBackend,
    events0::AbstractVector{<:AbstractMatrix{<:Real}};
    R::Real = 1.0,
    beta::Real = 1.0,
    norm::Bool = false,
    gdim::Union{Nothing,Int} = nothing,
    n_iter_max::Int = 100_000,
    metric::GroundMetric = EuclideanMetric(),
    strict::Bool = false,
)
    tuples0 = _unpack_events(backend, events0, gdim)

    return _pairwise_emd_self(
        backend, tuples0;
        beta, R, norm,
        max_iter=n_iter_max,
        symmetric=true,
        metric,
        strict,
    )
end

function _emds_backend(
    backend::NetworkSimplexBackend,
    events0::AbstractVector{<:AbstractMatrix{<:Real}},
    events1::AbstractVector{<:AbstractMatrix{<:Real}};
    R::Real = 1.0,
    beta::Real = 1.0,
    norm::Bool = false,
    gdim::Union{Nothing,Int} = nothing,
    n_iter_max::Int = 100_000,
    metric::GroundMetric = EuclideanMetric(),
    strict::Bool = false,
)
    tuples0 = _unpack_events(backend, events0, gdim)
    tuples1 = _unpack_events(backend, events1, gdim)

    return _pairwise_emd_cross(
        backend, tuples0, tuples1;
        beta, R, norm,
        max_iter=n_iter_max,
        metric,
        strict,
    )
end

function _emds_backend!(
    backend::NetworkSimplexBackend{V},
    results::AbstractVector{V},
    events0::AbstractVector{<:AbstractMatrix{<:Real}};
    R::Real = 1.0,
    beta::Real = 1.0,
    norm::Bool = false,
    gdim::Union{Nothing,Int} = nothing,
    n_iter_max::Int = 100_000,
    metric::GroundMetric = EuclideanMetric(),
    strict::Bool = false,
) where {V}
    tuples0 = _unpack_events(backend, events0, gdim)

    return _pairwise_emd_self!(
        backend, results, tuples0;
        beta, R, norm,
        max_iter=n_iter_max,
        metric,
        strict,
    )
end

# ─────────────────────────────────────────────────────────────────────
# emds_ns64 / emds_ns64! — pairwise NS Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ns64(events0, events1=nothing; R=1.0, beta=1.0, norm=false,
              gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric(),
              strict=false)

Compute pairwise EMDs using the Network Simplex Float64 backend.

# Arguments
- `events0`: Vector of event matrices (each M x (1+gdim)).
- `events1`: `nothing` for self-pairwise, or a second Vector of events for cross-pairwise.
- `R`, `beta`, `norm`, `gdim`, `n_iter_max`, `metric`, `strict`: same as
  [`emd_ns64`](@ref).

# Returns
- Self-pairwise (`events1=nothing`): `Vector{Float64}` of length `n*(n-1)/2`
  in SciPy pdist (upper-triangular row-major) order.
- Cross-pairwise: `Matrix{Float64}` of shape `(length(events0), length(events1))`.
"""
function emds_ns64(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                   events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                   R::Real         = 1.0,
                   beta::Real      = 1.0,
                   norm::Bool      = false,
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = EuclideanMetric(),
                   strict::Bool = false)
    if events1 === nothing
        return _emds_backend(
            NS64, events0;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    else
        return _emds_backend(
            NS64, events0, events1;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    end
end

"""
    emds_ns64!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000, metric=EuclideanMetric(), strict=false)

Writes pairwise EMD values to the preallocated `results` vector (flat
upper-triangular, length `n*(n-1)/2`). Internal worker workspaces are created
for each call.
"""
function emds_ns64!(results::AbstractVector{Float64},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric(),
                    strict::Bool = false)
    return _emds_backend!(
        NS64, results, events0;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end

# ─────────────────────────────────────────────────────────────────────
# emds_ot64 / emds_ot64! — pairwise OT-style (arc mixing) Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ot64(events0, events1=nothing; R=1.0, beta=1.0, norm=false,
              gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric(),
              strict=false)

Pairwise EMDs using the OT-style Float64 backend (arc mixing enabled).
See `emds_ns64` for argument documentation.
"""
function emds_ot64(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                   events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                   R::Real         = 1.0,
                   beta::Real      = 1.0,
                   norm::Bool      = false,
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = EuclideanMetric(),
                   strict::Bool = false)
    if events1 === nothing
        return _emds_backend(
            OT64, events0;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    else
        return _emds_backend(
            OT64, events0, events1;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    end
end

"""
    emds_ot64!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000, metric=EuclideanMetric(), strict=false)

In-place pairwise EMDs using the OT-style Float64 backend (arc mixing enabled).
See `emds_ns64!` for argument documentation.
"""
function emds_ot64!(results::AbstractVector{Float64},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric(),
                    strict::Bool = false)
    return _emds_backend!(
        OT64, results, events0;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end

# ─────────────────────────────────────────────────────────────────────
# emds_ns32 / emds_ns32! — pairwise NS Float32 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ns32(events0, events1=nothing; R=1.0, beta=1.0, norm=false,
              gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric(),
              strict=false)

Pairwise EMDs using the Network Simplex Float32 backend.
Returns `Vector{Float32}` (self) or `Matrix{Float32}` (cross).
"""
function emds_ns32(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                   events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                   R::Real         = 1.0,
                   beta::Real      = 1.0,
                   norm::Bool      = false,
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = EuclideanMetric(),
                   strict::Bool = false)
    if events1 === nothing
        return _emds_backend(
            NS32, events0;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    else
        return _emds_backend(
            NS32, events0, events1;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    end
end

"""
    emds_ns32!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000, metric=EuclideanMetric(), strict=false)

In-place pairwise EMDs using the Network Simplex Float32 backend.
"""
function emds_ns32!(results::AbstractVector{Float32},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric(),
                    strict::Bool = false)
    return _emds_backend!(
        NS32, results, events0;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end

# ─────────────────────────────────────────────────────────────────────
# emds_ot32 / emds_ot32! — pairwise OT-style Float32 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ot32(events0, events1=nothing; R=1.0, beta=1.0, norm=false,
              gdim=nothing, n_iter_max=100_000, metric=EuclideanMetric(),
              strict=false)

Pairwise EMDs using the OT-style Float32 backend (arc mixing enabled).
"""
function emds_ot32(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                   events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                   R::Real         = 1.0,
                   beta::Real      = 1.0,
                   norm::Bool      = false,
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = EuclideanMetric(),
                   strict::Bool = false)
    if events1 === nothing
        return _emds_backend(
            OT32, events0;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    else
        return _emds_backend(
            OT32, events0, events1;
            R, beta, norm, gdim, n_iter_max, metric, strict,
        )
    end
end

"""
    emds_ot32!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000, metric=EuclideanMetric(), strict=false)

In-place pairwise EMDs using the OT-style Float32 backend (arc mixing enabled).
"""
function emds_ot32!(results::AbstractVector{Float32},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric(),
                    strict::Bool = false)
    return _emds_backend!(
        OT32, results, events0;
        R, beta, norm, gdim, n_iter_max, metric, strict,
    )
end
