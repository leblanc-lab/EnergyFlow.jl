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
function _pairwise_emd_self(::Type{V}, events::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractMatrix}};
                            beta::Real = 1.0, R::Real = 1.0, norm::Bool = true,
                            max_iter::Int = 100_000, symmetric::Bool = true,
                            arc_mixing::Bool = false,
                            metric::GroundMetric = EuclideanMetric()) where {V<:AbstractFloat}
    nev = length(events)

    max_n = maximum(length(e[1]) for e in events)
    npairs = nev * (nev - 1) ÷ 2

    results = Vector{V}(undef, npairs)

    make_ws() = begin
        w = EMDWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm, metric=metric)
        w.parallel_threshold = typemax(Int)
        w
    end
    work!(ws, k) = begin
        i, j = _flat_to_pair(k, nev)

        w0, c0 = events[i]
        w1, c1 = events[j]

        val, _ = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        results[k] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)

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
function _pairwise_emd_cross(::Type{V}, events_a::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractMatrix}},
                             events_b::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractMatrix}};
                             beta::Real = 1.0, R::Real = 1.0, norm::Bool = true,
                             max_iter::Int = 100_000,
                             arc_mixing::Bool = false,
                             metric::GroundMetric = EuclideanMetric()) where {V<:AbstractFloat}
    na = length(events_a)
    nb = length(events_b)

    max_na = maximum(length(e[1]) for e in events_a)
    max_nb = maximum(length(e[1]) for e in events_b)
    max_n = max(max_na, max_nb)

    D = Matrix{V}(undef, na, nb)

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

        val, _ = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        D[i, j] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)

    return D
end

"""
In-place self-pairwise: writes to pre-allocated results vector.
"""
function _pairwise_emd_self!(results::AbstractVector{V},
                             events::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractMatrix}};
                             beta::Real = 1.0, R::Real = 1.0, norm::Bool = true,
                             max_iter::Int = 100_000,
                             arc_mixing::Bool = false,
                             metric::GroundMetric = EuclideanMetric()) where V

    nev = length(events)
    npairs = nev * (nev - 1) ÷ 2
    @assert length(results) >= npairs "results vector too short"

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

        val, _ = _emd_raw!(ws,
                           convert(Vector{V}, w0), convert(Matrix{V}, c0),
                           convert(Vector{V}, w1), convert(Matrix{V}, c1);
                           max_iter=max_iter, arc_mixing=arc_mixing)
        results[k] = val
    end
    _pairwise_parallel!(npairs, make_ws, work!)

    return results
end

# ─────────────────────────────────────────────────────────────────────
# emds_ns64 / emds_ns64! — pairwise NS Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ns64(events0, events1=nothing; R=1.0, beta=1.0, norm=false, gdim=nothing,
              n_iter_max=100_000)

Compute pairwise EMDs using the Network Simplex Float64 backend.

# Arguments
- `events0`: Vector of event matrices (each M x (1+gdim)).
- `events1`: `nothing` for self-pairwise, or a second Vector of events for cross-pairwise.
- `R`, `beta`, `norm`, `gdim`, `n_iter_max`: same as `emd_ns64`.

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
                   metric::GroundMetric = EuclideanMetric())

    # Unpack all events into (weights, coords) tuples
    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(evs, Ref(gdim)))]

    if events1 === nothing
        tuples0 = _unpack(events0)
        return _pairwise_emd_self(Float64, tuples0; beta=beta, R=R, norm=norm,
                                  max_iter=n_iter_max, symmetric=true, metric=metric)
    else
        tuples0 = _unpack(events0)
        tuples1 = _unpack(events1)
        return _pairwise_emd_cross(Float64, tuples0, tuples1; beta=beta, R=R, norm=norm,
                                   max_iter=n_iter_max, metric=metric)
    end
end

"""
    emds_ns64!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000)

In-place version: writes pairwise EMD values to pre-allocated `results` vector
(flat upper-triangular, length n*(n-1)/2).
"""
function emds_ns64!(results::AbstractVector{Float64},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(evs, Ref(gdim)))]
    tuples0 = _unpack(events0)
    return _pairwise_emd_self!(results, tuples0; beta=beta, R=R, norm=norm,
                               max_iter=n_iter_max, metric=metric)
end

# ─────────────────────────────────────────────────────────────────────
# emds_ot64 / emds_ot64! — pairwise OT-style (arc mixing) Float64 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ot64(events0, events1=nothing; R=1.0, beta=1.0, norm=false, gdim=nothing,
              n_iter_max=100_000)

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
                   metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(evs, Ref(gdim)))]

    if events1 === nothing
        tuples0 = _unpack(events0)
        return _pairwise_emd_self(Float64, tuples0; beta=beta, R=R, norm=norm,
                                  max_iter=n_iter_max, symmetric=true, arc_mixing=true, metric=metric)
    else
        tuples0 = _unpack(events0)
        tuples1 = _unpack(events1)
        return _pairwise_emd_cross(Float64, tuples0, tuples1; beta=beta, R=R, norm=norm,
                                   max_iter=n_iter_max, arc_mixing=true, metric=metric)
    end
end

"""
    emds_ot64!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000)

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
                    metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(evs, Ref(gdim)))]
    tuples0 = _unpack(events0)
    return _pairwise_emd_self!(results, tuples0; beta=beta, R=R, norm=norm,
                               max_iter=n_iter_max, arc_mixing=true, metric=metric)
end

# ─────────────────────────────────────────────────────────────────────
# emds_ns32 / emds_ns32! — pairwise NS Float32 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ns32(events0, events1=nothing; R=1.0, beta=1.0, norm=false, gdim=nothing,
              n_iter_max=100_000)

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
                   metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(Ref(Float32), evs, Ref(gdim)))]

    if events1 === nothing
        tuples0 = _unpack(events0)
        return _pairwise_emd_self(Float32, tuples0; beta=beta, R=R, norm=norm,
                                  max_iter=n_iter_max, symmetric=true, metric=metric)
    else
        tuples0 = _unpack(events0)
        tuples1 = _unpack(events1)
        return _pairwise_emd_cross(Float32, tuples0, tuples1; beta=beta, R=R, norm=norm,
                                   max_iter=n_iter_max, metric=metric)
    end
end

"""
    emds_ns32!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000)

In-place pairwise EMDs using the Network Simplex Float32 backend.
"""
function emds_ns32!(results::AbstractVector{Float32},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(Ref(Float32), evs, Ref(gdim)))]
    tuples0 = _unpack(events0)
    return _pairwise_emd_self!(results, tuples0; beta=beta, R=R, norm=norm,
                               max_iter=n_iter_max, metric=metric)
end

# ─────────────────────────────────────────────────────────────────────
# emds_ot32 / emds_ot32! — pairwise OT-style Float32 backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_ot32(events0, events1=nothing; R=1.0, beta=1.0, norm=false, gdim=nothing,
              n_iter_max=100_000)

Pairwise EMDs using the OT-style Float32 backend (arc mixing enabled).
"""
function emds_ot32(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                   events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                   R::Real         = 1.0,
                   beta::Real      = 1.0,
                   norm::Bool      = false,
                   gdim::Union{Nothing,Int} = nothing,
                   n_iter_max::Int = 100_000,
                   metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(Ref(Float32), evs, Ref(gdim)))]

    if events1 === nothing
        tuples0 = _unpack(events0)
        return _pairwise_emd_self(Float32, tuples0; beta=beta, R=R, norm=norm,
                                  max_iter=n_iter_max, symmetric=true, arc_mixing=true, metric=metric)
    else
        tuples0 = _unpack(events0)
        tuples1 = _unpack(events1)
        return _pairwise_emd_cross(Float32, tuples0, tuples1; beta=beta, R=R, norm=norm,
                                   max_iter=n_iter_max, arc_mixing=true, metric=metric)
    end
end

"""
    emds_ot32!(results, events0; R=1.0, beta=1.0, norm=false, gdim=nothing,
               n_iter_max=100_000)

In-place pairwise EMDs using the OT-style Float32 backend (arc mixing enabled).
"""
function emds_ot32!(results::AbstractVector{Float32},
                    events0::AbstractVector{<:AbstractMatrix{<:Real}};
                    R::Real         = 1.0,
                    beta::Real      = 1.0,
                    norm::Bool      = false,
                    gdim::Union{Nothing,Int} = nothing,
                    n_iter_max::Int = 100_000,
                    metric::GroundMetric = EuclideanMetric())

    _unpack(evs) = [(w, c) for (w, c) in (_unpack_event.(Ref(Float32), evs, Ref(gdim)))]
    tuples0 = _unpack(events0)
    return _pairwise_emd_self!(results, tuples0; beta=beta, R=R, norm=norm,
                               max_iter=n_iter_max, arc_mixing=true, metric=metric)
end
