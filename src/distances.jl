#=
    Distances: Cost matrix filling for EMD computation.

    Fills the cost matrix in the NetworkSimplex solver workspace.
    Cost of arc (i,j) = (distance(i,j) / R)^beta.
    Fictitious particle arcs get cost = 1.

    Provides serial, chunked, and parallel variants.
    Ground metric is selected via type dispatch on GroundMetric subtypes.
    (Type definitions are in GroundMetrics.jl, loaded before EMDSolver.jl.)
=#

# ─────────────────────────────────────────────────────────────────────
# Helper: fill fictitious particle costs
# ─────────────────────────────────────────────────────────────────────

function _fill_fict_costs!(ns, ::Type{V}, n0_eff::Int, n1_eff::Int,
                           has_fict_source::Bool, has_fict_target::Bool) where V
    if has_fict_source
        base = (n0_eff - 1) * n1_eff
        @inbounds for j in 1:n1_eff
            ns.costs[base + j] = one(V)
        end
    end
    if has_fict_target
        @inbounds for i in 1:n0_eff
            base = (i - 1) * n1_eff
            ns.costs[base + n1_eff] = one(V)
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
# Cost matrix filling (serial) — dispatched on GroundMetric
# ─────────────────────────────────────────────────────────────────────

"""
    _fill_costs!(ws, metric, coords0, coords1, n0, n1, has_fict_source, has_fict_target)

Fill the cost matrix in the NetworkSimplex solver (serial).
Dispatches on the `metric` type.
"""
function _fill_costs!(ws::EMDWorkspace{V}, ::EuclideanMetric,
                      coords0::AbstractMatrix{V},
                      coords1::AbstractMatrix{V},
                      n0_eff::Int, n1_eff::Int,
                      has_fict_source::Bool,
                      has_fict_target::Bool) where V
    ns = ws.ns
    beta = ws.beta
    R = ws.R
    R2 = R * R

    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff
    dim = size(coords0, 1)

    # Fill real particle distances
    @inbounds for i in 1:n0_real
        base = (i - 1) * n1_eff
        for j in 1:n1_real
            # Squared Euclidean distance
            d2 = zero(V)
            @simd for k in 1:dim
                diff = coords0[k, i] - coords1[k, j]
                d2 += diff * diff
            end

            # Apply beta and R
            if beta == V(1)
                cost = sqrt(d2) / R
            elseif beta == V(2)
                cost = d2 / R2
            else
                cost = (d2 / R2)^(beta / V(2))
            end

            ns.costs[base + j] = cost
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

function _fill_costs!(ws::EMDWorkspace{V}, ::SquaredEuclideanMetric,
                      coords0::AbstractMatrix{V},
                      coords1::AbstractMatrix{V},
                      n0_eff::Int, n1_eff::Int,
                      has_fict_source::Bool,
                      has_fict_target::Bool) where V
    ns = ws.ns
    beta = ws.beta
    R = ws.R
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff
    dim = size(coords0, 1)

    # SquaredEuclidean: ground distance = d² (no sqrt)
    # cost = (d² / R)^beta
    # Verify: SqEuc(beta=1, R=1) vs Euc(beta=2, R=1):
    #   SqEuc: cost = d²/1 = d²
    #   Euc:   cost = (sqrt(d²))²/1 = d²  ✓
    @inbounds for i in 1:n0_real
        base = (i - 1) * n1_eff
        for j in 1:n1_real
            d2 = zero(V)
            @simd for k in 1:dim
                diff = coords0[k, i] - coords1[k, j]
                d2 += diff * diff
            end

            cost = if beta == one(V)
                d2 / R
            elseif beta == V(2)
                (d2 / R)^2
            else
                (d2 / R)^beta
            end
            ns.costs[base + j] = cost
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

function _fill_costs!(ws::EMDWorkspace{V}, ::EtaPhiMetric,
                      coords0::AbstractMatrix{V},
                      coords1::AbstractMatrix{V},
                      n0_eff::Int, n1_eff::Int,
                      has_fict_source::Bool,
                      has_fict_target::Bool) where V
    ns = ws.ns
    beta = ws.beta
    R = ws.R
    R2 = R * R
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff

    TWO_PI = V(2π)
    PI = V(π)

    @inbounds for i in 1:n0_real
        base = (i - 1) * n1_eff
        eta0 = coords0[1, i]
        phi0 = coords0[2, i]
        for j in 1:n1_real
            deta = eta0 - coords1[1, j]
            dphi_raw = abs(phi0 - coords1[2, j])
            dphi = PI - abs(mod(dphi_raw, TWO_PI) - PI)
            d2 = deta * deta + dphi * dphi

            cost = if beta == one(V)
                sqrt(d2) / R
            elseif beta == V(2)
                d2 / R2
            else
                (d2 / R2)^(beta / V(2))
            end
            ns.costs[base + j] = cost
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

function _fill_costs!(ws::EMDWorkspace{V}, m::PrecomputedMetric,
                      coords0::AbstractMatrix{V},
                      coords1::AbstractMatrix{V},
                      n0_eff::Int, n1_eff::Int,
                      has_fict_source::Bool,
                      has_fict_target::Bool) where V
    ns = ws.ns
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff

    @inbounds for i in 1:n0_real
        base = (i - 1) * n1_eff
        for j in 1:n1_real
            ns.costs[base + j] = V(m.costs[i, j])
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

function _fill_costs!(ws::EMDWorkspace{V}, m::CustomMetric,
                      coords0::AbstractMatrix{V},
                      coords1::AbstractMatrix{V},
                      n0_eff::Int, n1_eff::Int,
                      has_fict_source::Bool,
                      has_fict_target::Bool) where V
    ns = ws.ns
    beta = ws.beta
    R = ws.R
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff

    @inbounds for i in 1:n0_real
        base = (i - 1) * n1_eff
        ci = @view coords0[:, i]
        for j in 1:n1_real
            cj = @view coords1[:, j]
            d = V(m.dist_fn(ci, cj))
            cost = (d / R)^beta
            ns.costs[base + j] = cost
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

"""
    _fill_costs_chunk!(costs, coords0, coords1, i_start, i_end, n1_eff, n1_real, dim, beta, R, R2)

Fill a contiguous chunk of source rows `i_start:i_end` in the cost matrix (Euclidean).
Used as the per-thread body of `_fill_costs_parallel!`.
Arguments are passed explicitly to avoid closure-capture issues in Julia 1.12.
"""
function _fill_costs_chunk!(::EuclideanMetric,
                             costs::Vector{V},
                             coords0::AbstractMatrix{V},
                             coords1::AbstractMatrix{V},
                             i_start::Int, i_end::Int,
                             n1_eff::Int, n1_real::Int, dim::Int,
                             beta::V, R::V, R2::V) where V
    for i in i_start:i_end
        base = (i - 1) * n1_eff
        @inbounds for j in 1:n1_real
            d2 = zero(V)
            @simd for k in 1:dim
                diff = coords0[k, i] - coords1[k, j]
                d2 += diff * diff
            end

            if beta == V(1)
                cost = sqrt(d2) / R
            elseif beta == V(2)
                cost = d2 / R2
            else
                cost = (d2 / R2)^(beta / V(2))
            end

            costs[base + j] = cost
        end
    end
    return nothing
end

function _fill_costs_chunk!(::SquaredEuclideanMetric,
                             costs::Vector{V},
                             coords0::AbstractMatrix{V},
                             coords1::AbstractMatrix{V},
                             i_start::Int, i_end::Int,
                             n1_eff::Int, n1_real::Int, dim::Int,
                             beta::V, R::V, R2::V) where V
    for i in i_start:i_end
        base = (i - 1) * n1_eff
        @inbounds for j in 1:n1_real
            d2 = zero(V)
            @simd for k in 1:dim
                diff = coords0[k, i] - coords1[k, j]
                d2 += diff * diff
            end

            cost = if beta == one(V)
                d2 / R
            elseif beta == V(2)
                (d2 / R)^2
            else
                (d2 / R)^beta
            end

            costs[base + j] = cost
        end
    end
    return nothing
end

function _fill_costs_chunk!(::EtaPhiMetric,
                             costs::Vector{V},
                             coords0::AbstractMatrix{V},
                             coords1::AbstractMatrix{V},
                             i_start::Int, i_end::Int,
                             n1_eff::Int, n1_real::Int, dim::Int,
                             beta::V, R::V, R2::V) where V
    TWO_PI = V(2π)
    PI = V(π)
    for i in i_start:i_end
        base = (i - 1) * n1_eff
        eta0 = coords0[1, i]
        phi0 = coords0[2, i]
        @inbounds for j in 1:n1_real
            deta = eta0 - coords1[1, j]
            dphi_raw = abs(phi0 - coords1[2, j])
            dphi = PI - abs(mod(dphi_raw, TWO_PI) - PI)
            d2 = deta * deta + dphi * dphi

            cost = if beta == one(V)
                sqrt(d2) / R
            elseif beta == V(2)
                d2 / R2
            else
                (d2 / R2)^(beta / V(2))
            end

            costs[base + j] = cost
        end
    end
    return nothing
end

"""
    _fill_costs_parallel!(ws, metric, coords0, coords1, n0_eff, n1_eff, has_fict_source, has_fict_target)

Parallel version of `_fill_costs!`. Parallelizes over source particles.
For PrecomputedMetric and CustomMetric, falls back to serial.
"""
function _fill_costs_parallel!(ws::EMDWorkspace{V}, metric::Union{EuclideanMetric, SquaredEuclideanMetric, EtaPhiMetric},
                                coords0::AbstractMatrix{V},
                                coords1::AbstractMatrix{V},
                                n0_eff::Int, n1_eff::Int,
                                has_fict_source::Bool,
                                has_fict_target::Bool) where V
    ns   = ws.ns
    beta = ws.beta
    R    = ws.R
    R2   = R * R

    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff
    dim     = size(coords0, 1)

    nt         = Threads.nthreads()
    chunk_size = max(1, (n0_real + nt - 1) ÷ nt)
    costs      = ns.costs

    @sync for t in 1:nt
        i_start = (t - 1) * chunk_size + 1
        i_end   = min(t * chunk_size, n0_real)
        if i_start <= n0_real
            Threads.@spawn _fill_costs_chunk!(metric, costs, coords0, coords1,
                                              i_start, i_end,
                                              n1_eff, n1_real, dim,
                                              beta, R, R2)
        end
    end

    _fill_fict_costs!(ns, V, n0_eff, n1_eff, has_fict_source, has_fict_target)
    return nothing
end

# Fallback: PrecomputedMetric and CustomMetric use serial
function _fill_costs_parallel!(ws::EMDWorkspace{V}, metric::GroundMetric,
                                coords0::AbstractMatrix{V},
                                coords1::AbstractMatrix{V},
                                n0_eff::Int, n1_eff::Int,
                                has_fict_source::Bool,
                                has_fict_target::Bool) where V
    _fill_costs!(ws, metric, coords0, coords1, n0_eff, n1_eff, has_fict_source, has_fict_target)
end
