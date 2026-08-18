#=
    Utils.jl

    Utils: Event format helpers for EMD computation.

    Provides _unpack_event for extracting weights and coordinates from
    EnergyFlow-format event matrices, and _emd_raw_alloc for convenience
    EMD computation with automatic workspace allocation.
=#

# ─────────────────────────────────────────────────────────────────────
# Event format helpers
# ─────────────────────────────────────────────────────────────────────

"""
    _unpack_event(::Type{V}, ev, gdim) -> (weights::Vector{V}, coords::Matrix{V})

Extract weights and coordinates from an EnergyFlow-format event matrix,
converting to element type `V`.

`ev` is an M×(1+d) matrix: column 1 is weights, columns 2:(1+d) are coords.
If `gdim` is not `nothing`, only the first `gdim` coordinate columns are used.
Returns `coords` as a `dim×M` matrix (column-major, one column per particle).
"""
function _unpack_event(::Type{V}, ev::AbstractMatrix{<:Real}, gdim::Union{Nothing,Int}) where {V<:AbstractFloat}
    M = size(ev, 1)
    ncols = size(ev, 2)

    # Determine number of coordinate dimensions
    d = ncols - 1  # all coordinate columns by default
    if gdim !== nothing
        if gdim > d
            error("gdim=$gdim but event only has $d coordinate column(s)")
        end
        d = gdim
    end

    weights = Vector{V}(undef, M)
    coords  = Matrix{V}(undef, d, M)   # dim × M

    @inbounds for i in 1:M
        weights[i] = V(ev[i, 1])
        for k in 1:d
            coords[k, i] = V(ev[i, k + 1])
        end
    end

    return weights, coords
end

# Default to Float64 for backward compatibility
_unpack_event(ev::AbstractMatrix{<:Real}, gdim::Union{Nothing,Int}) = _unpack_event(Float64, ev, gdim)

# ─────────────────────────────────────────────────────────────────────
# HepMC3 event loading
# ─────────────────────────────────────────────────────────────────────

"""
    load_hepmc3_events(filepath; maxevents=-1, skipevents=0, status=1, min_pt=0.0)
        -> Vector{Matrix{Float64}}

Load events from a HepMC3 ASCII file and convert to EnergyFlow event matrices.
Each event is an M×3 `Matrix{Float64}` with columns [pT, η, φ].

Only final-state particles with the given `status` code (default 1) are kept.
Particles with pT ≤ `min_pt` (including beam-axis particles with pT = 0) are
skipped. This is a standalone parser; it does not require `LorentzVectorHEP`.

# HepMC3 ASCII P-line format
`P barcode vertex pdgid px py pz energy mass status [...]`

# Coordinate conventions
- pT  = √(px² + py²)          — particle weight
- η   = asinh(pz / pT)        — pseudorapidity
- φ   = atan(py, px) ∈ (-π,π] — azimuthal angle (radians)
"""
function load_hepmc3_events(filepath::AbstractString;
                             maxevents::Int  = -1,
                             skipevents::Int = 0,
                             status::Int     = 1,
                             min_pt::Real    = 0.0)
    events   = Matrix{Float64}[]
    raw      = NTuple{3,Float64}[]   # reusable per-event buffer
    pt_cut   = Float64(min_pt)
    n_ev     = 0                     # events processed (after skip)
    n_seen   = 0                     # events encountered in file

    open(filepath) do f
        for line in eachline(f)
            # End-of-listing sentinel
            occursin(r"HepMC::.*-END_EVENT_LISTING", line) && break

            c = isempty(line) ? '\0' : line[1]

            if c == 'E' && length(line) > 1 && line[2] == ' '
                # Flush previous event (if we were collecting it)
                if n_seen > skipevents && !isempty(raw)
                    push!(events, _particles_to_ptEtaPhi(raw, pt_cut))
                    n_ev += 1
                end
                empty!(raw)
                n_seen += 1
                (maxevents >= 0 && n_ev >= maxevents) && break

            elseif c == 'P' && length(line) > 1 && line[2] == ' '
                n_seen <= skipevents && continue   # skipping this event
                tok = split(line)
                length(tok) < 10 && continue
                parse(Int, tok[10]) == status || continue
                px = parse(Float64, tok[5])
                py = parse(Float64, tok[6])
                pz = parse(Float64, tok[7])
                push!(raw, (px, py, pz))
            end
        end
        # Flush the last event if we were collecting it
        if n_seen > skipevents && !isempty(raw) && (maxevents < 0 || n_ev < maxevents)
            push!(events, _particles_to_ptEtaPhi(raw, pt_cut))
        end
    end

    return events
end

function _particles_to_ptEtaPhi(particles::Vector{NTuple{3,Float64}}, pt_cut::Float64)
    # Two-pass: count qualifying particles, then fill matrix
    n_keep = 0
    for (px, py, _) in particles
        sqrt(px * px + py * py) > pt_cut && (n_keep += 1)
    end
    ev = Matrix{Float64}(undef, n_keep, 3)
    row = 0
    @inbounds for (px, py, pz) in particles
        pt = sqrt(px * px + py * py)
        pt <= pt_cut && continue
        row += 1
        ev[row, 1] = pt
        ev[row, 2] = asinh(pz / pt)   # η = asinh(pz/pT), exact, no clamping needed
        ev[row, 3] = atan(py, px)     # φ ∈ (-π, π]
    end
    return ev
end

# ─────────────────────────────────────────────────────────────────────
# Convenience raw EMD with automatic workspace allocation
# ─────────────────────────────────────────────────────────────────────

"""
    _emd_raw_alloc(weights0, coords0, weights1, coords1; kwargs...) -> (V, Symbol)

Convenience raw EMD that allocates a workspace internally.
"""
function _emd_raw_alloc(weights0::AbstractVector{<:Real}, coords0::AbstractMatrix{<:Real},
                        weights1::AbstractVector{<:Real}, coords1::AbstractMatrix{<:Real};
                        beta::Real = 1.0, R::Real = 1.0, norm::Bool = true,
                        max_iter::Int = 100_000,
                        metric::GroundMetric = EuclideanMetric())
    V = promote_type(eltype(weights0), eltype(coords0), eltype(weights1), eltype(coords1), Float64)

    n0 = length(weights0)
    n1 = length(weights1)

    ws = EMDWorkspace{V}(n0, n1; beta=beta, R=R, norm=norm, metric=metric)

    w0 = convert(Vector{V}, weights0)
    c0 = convert(Matrix{V}, coords0)
    w1 = convert(Vector{V}, weights1)
    c1 = convert(Matrix{V}, coords1)

    return _emd_raw!(ws, w0, c0, w1, c1; max_iter=max_iter)
end

"""
    _transport_plan(ns::NetworkSimplexSolver; arc_mixing=false) -> Matrix

Reconstruct the optimal transport plan from a solved network-simplex workspace.
The returned matrix has size `ns.n0 × ns.n1`, where `ns.n0`/`ns.n1` are the
effective solver dimensions. For `norm=true` this is just the real-particle
transport block and the row/column sums are normalized to 1; for `norm=false`
it includes any fictitious source/target row or column introduced by the
solver and the returned plan is rescaled to the original physical weights.
The artificial root arcs are never included.
"""
function _transport_plan(ns::NetworkSimplexSolver{V}; arc_mixing::Bool = false) where V
    plan = Matrix{V}(undef, ns.n0, ns.n1)
    arc_num = ns.arc_num

    if arc_mixing
        perm = Vector{Int}(undef, arc_num)
        k = ns.block_size
        i = 1
        j = 1
        @inbounds for a_orig in 1:arc_num
            perm[a_orig] = i
            i += k
            if i > arc_num
                j += 1
                i = j
            end
        end

        @inbounds for a_orig in 1:arc_num
            a = perm[a_orig]
            plan[(a_orig - 1) ÷ ns.n1 + 1, (a_orig - 1) % ns.n1 + 1] = ns.flows[a]
        end
    else
        @inbounds for a in 1:arc_num
            plan[(a - 1) ÷ ns.n1 + 1, (a - 1) % ns.n1 + 1] = ns.flows[a]
        end
    end

    return plan
end
