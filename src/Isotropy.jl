#=
    Isotropy.jl

    Event isotropy helpers for EnergyFlow.jl.

    Provides reference geometries, normalized ground metrics, event selection
    helpers, and convenience front-door wrappers for the isotropy observable.
=#

wrap_dphi(a, b) = (d = abs(a - b); π - abs(mod(d, 2π) - π))

ring_event(ev) = ev[:, [1, 3]]

function sphere_event(ev)
    mask = vec(sum(abs2, @view(ev[:, 2:4]); dims=2)) .> 0
    return ev[mask, :]
end

function _frontdoor_event_view(event::AbstractMatrix{<:Real}, geometry::Symbol)
    if geometry === :ring
        return size(event, 2) == 2 ? event : ring_event(event)
    elseif geometry === :cylinder
        return event
    elseif geometry === :sphere
        return sphere_event(event)
    else
        error("Unknown isotropy geometry :$geometry. Available: :ring, :cylinder, :sphere")
    end
end


"""
    ring_reference(n) -> n×2 Matrix

Quasi-uniform ring reference event: `n` equal-weight particles at
`φ_j = 2π(j - 1/2)/n`. Columns are `[weight, φ]`.
"""
function ring_reference(n::Int)
    n > 0 || error("ring_reference(n) requires n > 0")
    hcat(fill(1.0 / n, n), [2π * (j - 0.5) / n for j in 1:n])
end

"""
    cylinder_reference(nphi, ymax) -> m×3 Matrix

Quasi-uniform cylinder reference event: `nphi` slices in `φ` and
`floor(ymax * nphi / π)` slices in `y` over `|y| ≤ ymax`. Columns are
`[weight, y, φ]`.
"""
function cylinder_reference(nphi::Int, ymax::Real)
    nphi > 0 || error("cylinder_reference(nphi, ymax) requires nphi > 0")
    ymax > 0 || error("cylinder_reference(nphi, ymax) requires ymax > 0")
    ny = floor(Int, ymax * nphi / π)
    ny > 0 || error("cylinder_reference(nphi, ymax) requires ymax*nphi/π ≥ 1")
    phis = [2π * (j - 0.5) / nphi for j in 1:nphi]
    ys = [-ymax + 2ymax * (i - 0.5) / ny for i in 1:ny]
    pts = [(y, phi) for phi in phis for y in ys]
    n = length(pts)
    hcat(fill(1.0 / n, n), first.(pts), last.(pts))
end

"""
    healpix_pix2vec_ring(nside) -> Vector{NTuple{3,Float64}}

Unit 3-vectors at the centers of the `12·nside²` HEALPix pixels in the RING
scheme. Reproduces `astropy_healpix.pix2vec` closely and is used internally by
[`sphere_reference`](@ref). `nside` must be positive.
"""
function healpix_pix2vec_ring(nside::Int)
    nside > 0 || error("healpix_pix2vec_ring(nside) requires nside > 0")
    npix = 12 * nside^2
    ncap = 2 * nside * (nside - 1)
    vecs = Vector{NTuple{3,Float64}}(undef, npix)
    for p in 0:npix-1
        if p < ncap
            ph = (p + 1) / 2
            i = floor(Int, sqrt(ph - sqrt(floor(ph)))) + 1
            j = p + 1 - 2 * i * (i - 1)
            z = 1.0 - i^2 / (3.0 * nside^2)
            phi = (π / (2i)) * (j - 0.5)
        elseif p < npix - ncap
            pp = p - ncap
            i = pp ÷ (4 * nside) + nside
            j = (pp % (4 * nside)) + 1
            s = (i - nside + 1) % 2
            z = 4.0 / 3.0 - 2.0 * i / (3.0 * nside)
            phi = (π / (2 * nside)) * (j - s / 2.0)
        else
            pp = npix - p
            ph = pp / 2
            i = floor(Int, sqrt(ph - sqrt(floor(ph)))) + 1
            j = 4 * i + 1 - (pp - 2 * i * (i - 1))
            z = -1.0 + i^2 / (3.0 * nside^2)
            phi = (π / (2i)) * (j - 0.5)
        end
        sth = sqrt(1 - z * z)
        vecs[p + 1] = (sth * cos(phi), sth * sin(phi), z)
    end
    return vecs
end

"""
    sphere_reference(nval) -> m×4 Matrix

Quasi-uniform spherical reference event built from the centers of the
`12·(2^nval)²` HEALPix pixels. Columns are `[weight, x, y, z]`.
"""
function sphere_reference(nval::Int)
    nval >= 0 || error("sphere_reference(nval) requires nval ≥ 0")
    vecs = healpix_pix2vec_ring(2^nval)
    n = length(vecs)
    hcat(fill(1.0 / n, n), getindex.(vecs, 1), getindex.(vecs, 2), getindex.(vecs, 3))
end


"""
    ring_cos_metric()

Ground distance on the ring, `(π/(π-2))·(1 - cos Δφ)`, with the normalization
baked in so isotropy uses `R=1`, `beta=1`, `norm=true`.
"""
ring_cos_metric() = CustomMetric((p, q) -> (π / (π - 2)) * (1 - cos(wrap_dphi(p[1], q[1]))))

"""
    ring_phi_metric()

Ground distance on the ring, `(4/π)·Δφ`, with the normalization baked in.
"""
ring_phi_metric() = CustomMetric((p, q) -> (4 / π) * wrap_dphi(p[1], q[1]))

"""
    cylinder_metric(ymax)

Ground distance on the cylinder, `12/(π² + 16·ymax²)·(Δy² + Δφ²)`, with the
normalization baked in.
"""
function cylinder_metric(ymax::Real)
    c = 12.0 / (π^2 + 16.0 * ymax^2)
    CustomMetric((p, q) -> begin
        dy = p[1] - q[1]
        dphi = wrap_dphi(p[2], q[2])
        c * (dy * dy + dphi * dphi)
    end)
end

_cosang(p, q) = clamp((p[1] * q[1] + p[2] * q[2] + p[3] * q[3]) /
                      (sqrt(p[1]^2 + p[2]^2 + p[3]^2) * sqrt(q[1]^2 + q[2]^2 + q[3]^2)),
                      -1.0, 1.0)

"""
    sphere_cos_metric()

Ground distance on the sphere, `2·(1 - cos θ)` between 3-momentum directions.
"""
sphere_cos_metric() = CustomMetric((p, q) -> 2.0 * (1.0 - _cosang(p, q)))

"""
    sphere_angular_metric()

Angular ground distance on the sphere, `arccos(cos θ)`.
"""
sphere_angular_metric() = CustomMetric((p, q) -> acos(_cosang(p, q)))


"""
    select_events(events, ymax; min_particles=2) -> Vector{Matrix}

Physics preselection for hadron-collider isotropy: keep particles with
`|y| ≤ ymax` and drop events with fewer than `min_particles` accepted
particles. This is a deliberate acceptance choice and should be applied
consistently when comparing implementations.
"""
select_events(events, ymax::Real; min_particles::Int=2) =
    [ev[abs.(ev[:, 2]) .<= ymax, :] for ev in events
     if count(abs.(ev[:, 2]) .<= ymax) >= min_particles]

"""
    select_sphere_events(events; min_particles=2) -> Vector{Matrix}

Physics preselection for spherical isotropy: drop zero-momentum particles and
any event with fewer than `min_particles` remaining. This is the acceptance
choice for the lepton-collider version of the observable.
"""
select_sphere_events(events; min_particles::Int=2) =
    [se for ev in events for se in (sphere_event(ev),) if size(se, 1) >= min_particles]

"""
    load_hepmc3_momenta(filepath; maxevents=-1, status=1) -> Vector{Matrix{Float64}}

Load events from a HepMC3 ASCII file as `M×4` matrices with columns
`[E, px, py, pz]`, suitable for spherical isotropy.
"""
function load_hepmc3_momenta(filepath::String; maxevents::Int=-1, status::Int=1)
    events = Matrix{Float64}[]
    raw = NTuple{4,Float64}[]
    n_ev = 0
    open(filepath) do f
        for line in eachline(f)
            occursin(r"HepMC::.*-END_EVENT_LISTING", line) && break
            c = isempty(line) ? '\0' : line[1]
            if c == 'E' && length(line) > 1 && line[2] == ' '
                if !isempty(raw)
                    push!(events, reduce(vcat, (reshape(collect(r), 1, 4) for r in raw)))
                    n_ev += 1
                end
                empty!(raw)
                (maxevents >= 0 && n_ev >= maxevents) && break
            elseif c == 'P' && length(line) > 1 && line[2] == ' '
                tok = split(line)
                length(tok) < 10 && continue
                parse(Int, tok[10]) == status || continue
                px = parse(Float64, tok[5]); py = parse(Float64, tok[6])
                pz = parse(Float64, tok[7]); e = parse(Float64, tok[8])
                push!(raw, (e, px, py, pz))
            end
        end
        if !isempty(raw) && (maxevents < 0 || n_ev < maxevents)
            push!(events, reduce(vcat, (reshape(collect(r), 1, 4) for r in raw)))
        end
    end
    return events
end

function _recoil_correct_ring(event::AbstractMatrix{<:Real})
    qx = sum(event[:, 1] .* cos.(event[:, 2]))
    qy = sum(event[:, 1] .* sin.(event[:, 2]))
    mag = hypot(qx, qy)
    mag == 0 && return event
    recoil = reshape([mag, atan(-qy, -qx)], 1, 2)
    return vcat(event, recoil)
end

function _recoil_correct_cylinder(event::AbstractMatrix{<:Real})
    qx = sum(event[:, 1] .* cos.(event[:, 3]))
    qy = sum(event[:, 1] .* sin.(event[:, 3]))
    mag = hypot(qx, qy)
    mag == 0 && return event
    recoil = reshape([mag, 0.0, atan(-qy, -qx)], 1, 3)
    return vcat(event, recoil)
end

function _recoil_correct_sphere(event::AbstractMatrix{<:Real})
    px = sum(event[:, 2])
    py = sum(event[:, 3])
    pz = sum(event[:, 4])
    mag = sqrt(px * px + py * py + pz * pz)
    mag == 0 && return event
    recoil = reshape([mag, -px, -py, -pz], 1, 4)
    return vcat(event, recoil)
end

function _maybe_recoil_correct(event::AbstractMatrix{<:Real}, ref::AbstractMatrix{<:Real}, recoil::Bool)
    recoil || return event
    ncols = size(ref, 2)
    if ncols == 2
        return _recoil_correct_ring(event)
    elseif ncols == 3
        return _recoil_correct_cylinder(event)
    elseif ncols == 4
        return _recoil_correct_sphere(event)
    else
        return event
    end
end


"""
    event_isotropy(event, ref, metric; backend=:ns64, n_iter_max=100_000,
                   recoil=false)

Compute event isotropy as the EMD between `event` and a quasi-uniform
reference event `ref` using the supplied ground `metric`. The isotropy
normalization is baked into the reference and metric, so the EMD is evaluated
with `R=1`, `beta=1`, `norm=true`. If `recoil=true`, a single pseudo-particle
is added to balance the event's net momentum before evaluating the EMD.
"""
function event_isotropy(event, ref, metric; backend::Symbol=:ns64, n_iter_max::Int=100_000,
                         recoil::Bool=false)
    event_corr = _maybe_recoil_correct(event, ref, recoil)
    emd(event_corr, ref; R=1.0, beta=1.0, norm=true, metric=metric,
        backend=backend, n_iter_max=n_iter_max)
end

"""
    event_isotropy(event; geometry=:ring, n=128, nphi=16, ymax=4.0, nval=2,
                   backend=:ns64, n_iter_max=100_000, recoil=false)

Convenience wrapper that builds the standard isotropy reference and metric for
the requested geometry and then computes the observable.

Supported geometries:
- `:ring`     — hadron-collider ring in `φ` (`n` points)
- `:cylinder` — hadron-collider cylinder in `(y, φ)` (`nphi × floor(ymax*nphi/π)` grid)
- `:sphere`   — lepton-collider sphere via HEALPix (`nval`, so `nside = 2^nval`)

For `:ring`, 3-column hadron-collider events are projected to `[pT, φ]`.
For `:sphere`, zero-momentum particles are dropped before evaluation.
Set `recoil=true` to append a balancing pseudo-particle before the EMD is
computed, which is useful when the input event has residual net momentum.
"""
function event_isotropy(event::AbstractMatrix{<:Real};
                        geometry::Symbol=:ring,
                        n::Int=128,
                        nphi::Int=16,
                        ymax::Real=4.0,
                        nval::Int=2,
                        backend::Symbol=:ns64,
                        n_iter_max::Int=100_000,
                        recoil::Bool=false)
    if geometry === :ring
        ev = _frontdoor_event_view(event, :ring)
        ref = ring_reference(n)
        return event_isotropy(_maybe_recoil_correct(ev, ref, recoil), ref, ring_cos_metric();
                              backend=backend, n_iter_max=n_iter_max)
    elseif geometry === :cylinder
        ev = _frontdoor_event_view(event, :cylinder)
        ref = cylinder_reference(nphi, ymax)
        return event_isotropy(_maybe_recoil_correct(ev, ref, recoil), ref, cylinder_metric(ymax);
                              backend=backend, n_iter_max=n_iter_max)
    elseif geometry === :sphere
        ev = _frontdoor_event_view(event, :sphere)
        ref = sphere_reference(nval)
        return event_isotropy(_maybe_recoil_correct(ev, ref, recoil), ref, sphere_cos_metric();
                              backend=backend, n_iter_max=n_iter_max)
    else
        error("Unknown isotropy geometry :$geometry. Available: :ring, :cylinder, :sphere")
    end
end