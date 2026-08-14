#=  
    Sinkhorn.jl

    Sinkhorn algorithm for entropy-regularized optimal transport.

    Implements log-domain Sinkhorn with ε-annealing for numerical stability.
    Provides both workspace-reusing and allocating API functions that match
    the NSSolver.jl patterns (emd_sinkhorn! / emd_sinkhorn).

    NOTE: Sinkhorn solves the *regularised* OT problem:
        OT_ε(a,b,C) = min_{P ∈ U(a,b)} ⟨C,P⟩ + ε H(P)
    The returned cost includes an entropic bias that shrinks as ε → 0
    but never vanishes for ε > 0.  For exact EMD use the NS backends.
=#

# ─────────────────────────────────────────────────────────────────────
# SinkhornWorkspace — pre-allocated workspace
# ─────────────────────────────────────────────────────────────────────

"""
    SinkhornWorkspace{V<:AbstractFloat}
    SinkhornWorkspace(max_n0, max_n1; beta=1.0, R=1.0, norm=true,
                      epsilon=0.01, max_iter=5000, tol=1e-9, annealing=true)

Pre-allocated workspace for approximate EMD computation with the Sinkhorn
(entropy-regularised OT) backend. Pass to [`emd!`](@ref) or `emd_sinkhorn!`.
The unparameterized constructor defaults to `Float64`.

# Arguments
- `max_n0`, `max_n1`: maximum particle counts of the two events in any
  subsequent solve.

# Keywords
- `beta`, `R`, `norm`: EMD parameters; see [`emd`](@ref).
- `epsilon`: entropic regularisation strength. Smaller values approximate the
  exact EMD more closely but require more iterations.
- `max_iter`: maximum Sinkhorn iterations per ε-level.
- `tol`: convergence tolerance on the marginal violation.
- `annealing`: if `true` (default), start from a large ε and geometrically
  anneal down to `epsilon`, warm-starting each stage — usually faster and
  more stable than solving directly at small ε.

!!! note
    Sinkhorn solves the *regularised* transport problem, so the returned cost
    carries an entropic bias that shrinks with `epsilon` but never fully
    vanishes. Use the network-simplex backends for exact values.

After a solve, the fields `total_cost`, `n_iters` and `converged` hold
diagnostics for the last computation.
"""
mutable struct SinkhornWorkspace{V<:AbstractFloat}
    # Parameters
    beta::V
    R::V
    norm::Bool
    epsilon::V           # regularization parameter
    max_iter::Int        # maximum Sinkhorn iterations per ε-level
    tol::V               # convergence tolerance
    annealing::Bool      # whether to use ε-annealing

    # Pre-allocated arrays
    max_n0::Int
    max_n1::Int
    C::Matrix{V}         # cost matrix (max_n0+1 × max_n1+1)
    log_K::Matrix{V}     # log kernel (max_n0+1 × max_n1+1)
    log_u::Vector{V}     # log scaling (max_n0+1)
    log_v::Vector{V}     # log scaling (max_n1+1)
    log_a::Vector{V}     # log source marginal
    log_b::Vector{V}     # log target marginal

    # Weight buffers (with room for fictitious particle)
    source_weights::Vector{V}
    target_weights::Vector{V}

    # Results
    total_cost::V
    n_iters::Int
    converged::Bool
end

function SinkhornWorkspace{V}(max_n0::Int, max_n1::Int;
                               beta::Real=1.0, R::Real=1.0, norm::Bool=true,
                               epsilon::Real=0.01, max_iter::Int=5000,
                               tol::Real=1e-9, annealing::Bool=true) where V
    SinkhornWorkspace{V}(
        V(beta), V(R), norm,
        V(epsilon), max_iter, V(tol), annealing,
        max_n0, max_n1,
        Matrix{V}(undef, max_n0 + 1, max_n1 + 1),
        Matrix{V}(undef, max_n0 + 1, max_n1 + 1),
        Vector{V}(undef, max_n0 + 1),
        Vector{V}(undef, max_n1 + 1),
        Vector{V}(undef, max_n0 + 1),
        Vector{V}(undef, max_n1 + 1),
        Vector{V}(undef, max_n0 + 1),
        Vector{V}(undef, max_n1 + 1),
        zero(V), 0, false
    )
end

SinkhornWorkspace(max_n0::Int, max_n1::Int; kwargs...) =
    SinkhornWorkspace{Float64}(max_n0, max_n1; kwargs...)

# ─────────────────────────────────────────────────────────────────────
# Log-domain Sinkhorn solver
# ─────────────────────────────────────────────────────────────────────

"""
    _sinkhorn_log_domain!(ws, n0, n1, epsilon) -> (cost, status)

Log-domain Sinkhorn iterations for a single epsilon value.
Weights must already be in `ws.source_weights` and `ws.target_weights`.
Cost matrix must already be filled in `ws.C[1:n0, 1:n1]`.
`ws.log_u` / `ws.log_v` must be initialized by the caller (zero or warm-start).
"""
function _sinkhorn_log_domain!(ws::SinkhornWorkspace{V},
                                n0::Int, n1::Int,
                                epsilon::V) where V
    # Build log-kernel from cost matrix
    inv_eps = one(V) / epsilon
    @inbounds for j in 1:n1, i in 1:n0
        ws.log_K[i,j] = -ws.C[i,j] * inv_eps
    end

    # Cache log-marginals (reuse ws buffers, no allocation)
    @inbounds for i in 1:n0
        ws.log_a[i] = log(max(ws.source_weights[i], V(1e-300)))
    end
    @inbounds for j in 1:n1
        ws.log_b[j] = log(max(ws.target_weights[j], V(1e-300)))
    end

    converged = false
    iter = 0

    while iter < ws.max_iter
        iter += 1

        # Log-domain u update: log_u[i] = log_a[i] - logsumexp_j(log_K[i,j] + log_v[j])
        @inbounds for i in 1:n0
            max_val = typemin(V)
            for j in 1:n1
                val = ws.log_K[i,j] + ws.log_v[j]
                if val > max_val
                    max_val = val
                end
            end
            sum_exp = zero(V)
            for j in 1:n1
                sum_exp += exp(ws.log_K[i,j] + ws.log_v[j] - max_val)
            end
            ws.log_u[i] = ws.log_a[i] - max_val - log(sum_exp)
        end

        # Log-domain v update: log_v[j] = log_b[j] - logsumexp_i(log_K[i,j] + log_u[i])
        @inbounds for j in 1:n1
            max_val = typemin(V)
            for i in 1:n0
                val = ws.log_K[i,j] + ws.log_u[i]
                if val > max_val
                    max_val = val
                end
            end
            sum_exp = zero(V)
            for i in 1:n0
                sum_exp += exp(ws.log_K[i,j] + ws.log_u[i] - max_val)
            end
            ws.log_v[j] = ws.log_b[j] - max_val - log(sum_exp)
        end

        # Check convergence every 10 iterations: max change in dual variables
        if iter % 10 == 0
            # Check marginal violation on rows
            max_err = zero(V)
            @inbounds for i in 1:n0
                max_val = typemin(V)
                for j in 1:n1
                    val = ws.log_u[i] + ws.log_K[i,j] + ws.log_v[j]
                    if val > max_val
                        max_val = val
                    end
                end
                row_sum = zero(V)
                for j in 1:n1
                    row_sum += exp(ws.log_u[i] + ws.log_K[i,j] + ws.log_v[j] - max_val)
                end
                row_sum *= exp(max_val)
                err = abs(row_sum - ws.source_weights[i])
                if err > max_err
                    max_err = err
                end
            end
            if max_err < ws.tol
                converged = true
                break
            end
        end
    end

    # Compute total cost: sum_ij C[i,j] * P[i,j]
    # where P[i,j] = exp(log_u[i] + log_K[i,j] + log_v[j])
    total = zero(V)
    @inbounds for j in 1:n1, i in 1:n0
        log_P = ws.log_u[i] + ws.log_K[i,j] + ws.log_v[j]
        if log_P > V(-500)
            total += ws.C[i,j] * exp(log_P)
        end
    end

    ws.total_cost = total
    ws.n_iters = iter
    ws.converged = converged

    return total, converged ? :optimal : :max_iter
end

function _sinkhorn_transport_plan(ws::SinkhornWorkspace{V}, n0::Int, n1::Int) where V
    plan = Matrix{V}(undef, n0, n1)
    @inbounds for j in 1:n1, i in 1:n0
        log_P = ws.log_u[i] + ws.log_K[i,j] + ws.log_v[j]
        plan[i, j] = log_P > V(-500) ? exp(log_P) : zero(V)
    end
    return plan
end

function _sinkhorn_plan_dims(ws::SinkhornWorkspace{V},
                             weights0::AbstractVector{V}, weights1::AbstractVector{V}) where V
    n0_eff = length(weights0)
    n1_eff = length(weights1)

    if !ws.norm
        weight_diff = sum(weights1) - sum(weights0)
        if weight_diff > zero(V)
            n0_eff += 1
        elseif weight_diff < zero(V)
            n1_eff += 1
        end
    end

    return n0_eff, n1_eff
end

# ─────────────────────────────────────────────────────────────────────
# Sinkhorn with ε-annealing
# ─────────────────────────────────────────────────────────────────────

"""
    _sinkhorn_annealing!(ws, n0, n1, target_eps) -> (cost, status)

Run Sinkhorn with ε-annealing: start from a large ε and geometrically
decrease to `target_eps`, warm-starting each stage from the previous solution.
"""
function _sinkhorn_annealing!(ws::SinkhornWorkspace{V},
                               n0::Int, n1::Int,
                               target_eps::V) where V
    # Annealing schedule: start from max(1.0, 64*target_eps) and halve
    eps_start = max(one(V), V(64) * target_eps)
    eps_current = eps_start

    # Save original settings
    orig_max_iter = ws.max_iter
    orig_tol = ws.tol

    total = zero(V)
    status = :max_iter
    total_iters = 0

    # Zero-init duals before first ε level
    fill!(@view(ws.log_u[1:n0]), zero(V))
    fill!(@view(ws.log_v[1:n1]), zero(V))

    done = false
    while !done
        if eps_current <= target_eps
            eps_current = target_eps
            done = true
        end

        # Use relaxed tolerance and fewer iters for intermediate steps
        if !done
            ws.max_iter = min(300, orig_max_iter)
            ws.tol = orig_tol * V(100)
        else
            ws.max_iter = orig_max_iter
            ws.tol = orig_tol
        end

        total, status = _sinkhorn_log_domain!(ws, n0, n1, eps_current)
        total_iters += ws.n_iters

        eps_current /= V(2)
    end

    # Restore settings
    ws.max_iter = orig_max_iter
    ws.tol = orig_tol
    ws.n_iters = total_iters

    return total, status
end

# ─────────────────────────────────────────────────────────────────────
# Raw EMD computation (weights + coords interface)
# ─────────────────────────────────────────────────────────────────────

"""
    _emd_sinkhorn_raw!(ws, weights0, coords0, weights1, coords1) -> (V, Symbol)

Compute EMD between two distributions using Sinkhorn with pre-allocated workspace.
Handles weight normalization and fictitious particles (for unbalanced case).
"""
function _emd_sinkhorn_raw!(ws::SinkhornWorkspace{V},
                             weights0::AbstractVector{V}, coords0::AbstractMatrix{V},
                             weights1::AbstractVector{V}, coords1::AbstractMatrix{V}) where V
    n0 = length(weights0)
    n1 = length(weights1)
    total0 = sum(weights0)
    total1 = sum(weights1)

    beta = ws.beta
    R = ws.R
    R2 = R * R
    dim = size(coords0, 1)

    # Handle weight balancing (Sinkhorn needs balanced OT)
    has_fict_source = false
    has_fict_target = false
    scale = one(V)

    if ws.norm
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
            has_fict_source = true
            n0_eff = n0 + 1
            n1_eff = n1
        elseif weight_diff < zero(V)
            has_fict_target = true
            n0_eff = n0
            n1_eff = n1 + 1
        else
            n0_eff = n0
            n1_eff = n1
        end

        max_total = max(total0, total1)
        scale = max_total

        @inbounds for i in 1:n0
            ws.source_weights[i] = weights0[i] / max_total
        end
        @inbounds for j in 1:n1
            ws.target_weights[j] = weights1[j] / max_total
        end

        if has_fict_source
            ws.source_weights[n0_eff] = abs(weight_diff) / max_total
        end
        if has_fict_target
            ws.target_weights[n1_eff] = abs(weight_diff) / max_total
        end
    end

    # Fill cost matrix
    n0_real = has_fict_source ? n0_eff - 1 : n0_eff
    n1_real = has_fict_target ? n1_eff - 1 : n1_eff

    @inbounds for i in 1:n0_real
        for j in 1:n1_real
            d2 = zero(V)
            @simd for k in 1:dim
                diff = coords0[k, i] - coords1[k, j]
                d2 += diff * diff
            end
            if beta == V(1)
                ws.C[i,j] = sqrt(d2) / R
            elseif beta == V(2)
                ws.C[i,j] = d2 / R2
            else
                ws.C[i,j] = (d2 / R2)^(beta / V(2))
            end
        end
    end

    # Fictitious particle costs = 1
    if has_fict_source
        @inbounds for j in 1:n1_eff
            ws.C[n0_eff, j] = one(V)
        end
    end
    if has_fict_target
        @inbounds for i in 1:n0_eff
            ws.C[i, n1_eff] = one(V)
        end
    end

    # Solve
    if ws.annealing
        emd_val, status = _sinkhorn_annealing!(ws, n0_eff, n1_eff, ws.epsilon)
    else
        # Zero-init duals for non-annealing path
        fill!(@view(ws.log_u[1:n0_eff]), zero(V))
        fill!(@view(ws.log_v[1:n1_eff]), zero(V))
        emd_val, status = _sinkhorn_log_domain!(ws, n0_eff, n1_eff, ws.epsilon)
    end

    if !ws.norm
        emd_val *= scale
    end

    if status !== :optimal
        # A non-optimal status may still produce a finite approximate transport
        # value, especially under warm-start annealing or early stopping. Keep
        # that value for non-strict callers, but reject truly invalid cases like
        # zero-iteration runs or non-finite results.
        if !isfinite(emd_val) || ws.max_iter <= 0
            return V(NaN), status
        end
        return emd_val, status
    end

    return emd_val, status
end

# ─────────────────────────────────────────────────────────────────────
# emd_sinkhorn! / emd_sinkhorn — single-pair Sinkhorn backend
# ─────────────────────────────────────────────────────────────────────

"""
    emd_sinkhorn!(ws::SinkhornWorkspace, ev0, ev1; gdim=nothing)

Compute an approximate EMD using the Sinkhorn backend with a pre-allocated
[`SinkhornWorkspace`](@ref). All Sinkhorn parameters (`beta`, `R`, `norm`,
`epsilon`, `tol`, `annealing`, `max_iter`) are read from the workspace.

Returns the regularised transport cost (same precision as the workspace).
"""
function emd_sinkhorn!(ws::SinkhornWorkspace{V},
                       ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                       gdim::Union{Nothing,Int} = nothing,
                       return_flow::Bool = false,
                       strict::Bool = false) where V

    w0, c0 = _unpack_event(V, ev0, gdim)
    w1, c1 = _unpack_event(V, ev1, gdim)

    n0_eff, n1_eff = _sinkhorn_plan_dims(ws, w0, w1)
    val, status = _emd_sinkhorn_raw!(ws, w0, c0, w1, c1)
    _handle_solver_status(status; strict=strict, backend=:sinkhorn, context="emd_sinkhorn!", value=val)
    return return_flow ? (val, _sinkhorn_transport_plan(ws, n0_eff, n1_eff)) : val
end

"""
    emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=false, gdim=nothing,
                 epsilon=0.01, max_iter_sinkhorn=5000, sinkhorn_tol=1e-9,
                 annealing=true) -> Float64

Compute an approximate EMD between two events using the Sinkhorn
(entropy-regularised OT) solver. Allocates a fresh workspace each call; for
repeated calls prefer [`emd_sinkhorn!`](@ref).

# Keywords
- `R`, `beta`, `norm`, `gdim`: as in [`emd`](@ref).
- `epsilon`: entropic regularisation strength (smaller = closer to exact EMD,
  slower to converge).
- `max_iter_sinkhorn`: maximum Sinkhorn iterations per ε-level.
- `sinkhorn_tol`: convergence tolerance on the marginal violation.
- `annealing`: use ε-annealing with warm starts (default `true`).

The returned value includes an entropic bias relative to the exact EMD; see
[`SinkhornWorkspace`](@ref).
"""
function emd_sinkhorn(ev0::AbstractMatrix{<:Real}, ev1::AbstractMatrix{<:Real};
                      R::Real        = 1.0,
                      beta::Real     = 1.0,
                      norm::Bool     = false,
                      gdim::Union{Nothing,Int} = nothing,
                      epsilon::Real  = 0.01,
                      max_iter_sinkhorn::Int = 5000,
                      sinkhorn_tol::Real = 1e-9,
                      annealing::Bool = true,
                      n_iter_max::Int = 100_000,  # accepted but unused (API compat)
                      return_flow::Bool = false,
                      strict::Bool = false)

    V = Float64
    w0, c0 = _unpack_event(V, ev0, gdim)
    w1, c1 = _unpack_event(V, ev1, gdim)

    n0 = length(w0)
    n1 = length(w1)
    ws = SinkhornWorkspace{V}(n0, n1; beta=beta, R=R, norm=norm,
                               epsilon=epsilon, max_iter=max_iter_sinkhorn,
                               tol=sinkhorn_tol, annealing=annealing)

    n0_eff, n1_eff = _sinkhorn_plan_dims(ws, w0, w1)
    val, status = _emd_sinkhorn_raw!(ws, w0, c0, w1, c1)
    _handle_solver_status(status; strict=strict, backend=:sinkhorn, context="emd_sinkhorn", value=val)
    return return_flow ? (val, _sinkhorn_transport_plan(ws, n0_eff, n1_eff)) : val
end

# ─────────────────────────────────────────────────────────────────────
# emds_sinkhorn — pairwise Sinkhorn backend
# ─────────────────────────────────────────────────────────────────────

"""
    emds_sinkhorn(events0, events1=nothing; R=1.0, beta=1.0, norm=false,
                  gdim=nothing, epsilon=0.01, max_iter_sinkhorn=5000,
                  sinkhorn_tol=1e-9, annealing=true)

Pairwise approximate EMDs using the Sinkhorn backend, multithreaded over
pairs. Return conventions match [`emds`](@ref): a flat upper-triangular
`Vector{Float64}` for self-pairwise mode (`events1 === nothing`), or a
`Matrix{Float64}` for cross-pairwise mode.

Sinkhorn-specific keywords are as in [`emd_sinkhorn`](@ref).
"""
function emds_sinkhorn(events0::AbstractVector{<:AbstractMatrix{<:Real}},
                       events1::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Real}}} = nothing;
                       R::Real         = 1.0,
                       beta::Real      = 1.0,
                       norm::Bool      = false,
                       gdim::Union{Nothing,Int} = nothing,
                       epsilon::Real   = 0.01,
                       max_iter_sinkhorn::Int = 5000,
                       sinkhorn_tol::Real = 1e-9,
                       annealing::Bool = true,
                       n_iter_max::Int = 100_000,
                       strict::Bool = false,
                       metric::GroundMetric = EuclideanMetric())

    if metric isa GroundMetric && !(metric isa EuclideanMetric)
        error("sinkhorn backend currently supports only EuclideanMetric().")
    end

    V = Float64
    _unpack(evs) = [_unpack_event(V, ev, gdim) for ev in evs]
    tuples0 = _unpack(events0)

    # Find max sizes
    max_n0 = maximum(length(t[1]) for t in tuples0)
    max_n1 = max_n0

    if events1 !== nothing
        tuples1 = _unpack(events1)
        max_n1 = max(max_n1, maximum(length(t[1]) for t in tuples1))
    end

    max_n = max(max_n0, max_n1)

    if events1 === nothing
        n = length(tuples0)
        npairs = n * (n - 1) ÷ 2
        results = Vector{V}(undef, npairs)
        statuses = Vector{Symbol}(undef, npairs)

        make_ws_self() = SinkhornWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm,
                                              epsilon=epsilon, max_iter=max_iter_sinkhorn,
                                              tol=sinkhorn_tol, annealing=annealing)
        work_self!(ws, k) = begin
            i, j = _flat_to_pair(k, n)
            w0, c0 = tuples0[i]
            w1, c1 = tuples0[j]

            val, status = _emd_sinkhorn_raw!(ws, w0, c0, w1, c1)
            statuses[k] = status
            results[k] = val
        end
        _pairwise_parallel!(npairs, make_ws_self, work_self!)
        _handle_pairwise_statuses(statuses; strict=strict, backend=:sinkhorn, context="emds_sinkhorn self pair")
        return results
    else
        tuples1 = _unpack(events1)
        na = length(tuples0)
        nb = length(tuples1)
        D = Matrix{V}(undef, na, nb)
        statuses = Vector{Symbol}(undef, na * nb)

        npairs = na * nb
        make_ws_cross() = SinkhornWorkspace{V}(max_n, max_n; beta=beta, R=R, norm=norm,
                                               epsilon=epsilon, max_iter=max_iter_sinkhorn,
                                               tol=sinkhorn_tol, annealing=annealing)
        work_cross!(ws, k) = begin
            i = (k - 1) ÷ nb + 1
            j = mod1(k, nb)
            w0, c0 = tuples0[i]
            w1, c1 = tuples1[j]

            val, status = _emd_sinkhorn_raw!(ws, w0, c0, w1, c1)
            statuses[k] = status
            D[i, j] = val
        end
        _pairwise_parallel!(npairs, make_ws_cross, work_cross!)
        _handle_pairwise_statuses(statuses; strict=strict, backend=:sinkhorn, context="emds_sinkhorn cross pair")
        return D
    end
end
