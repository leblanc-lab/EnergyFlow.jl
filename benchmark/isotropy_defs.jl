# Event isotropy definitions (Cesarotti & Thaler, arXiv:2004.06125)
#
# Event isotropy is the EMD between a (pT-normalized) event and a
# quasi-uniform reference event on a ring or cylinder. The ground distances
# and reference-event construction below follow the reference implementation
# (github.com/caricesarotti/event_isotropy, built on POT), so results are
# directly comparable to `ot.lp.emd2`.
#
# Conventions (see also the ATLAS measurement, arXiv:2305.16930):
#   ring:     N points uniform in φ, distance (π/(π-2))·(1 - cos Δφ)
#   cylinder: n_φ × n_y grid over |y| ≤ ymax, distance
#             12/(π² + 16·ymax²) · (Δy² + Δφ²), φ wrapped
#   sphere:   HEALPix-tiled unit sphere (12·nside² points), angular distance
#             on 3-momentum directions — the e⁺e⁻ / lepton-collider version
#
# The ring and cylinder metrics bake their normalization into the ground
# distance; the sphere uses the library's raw cos/angular measures. Use all
# with R=1, beta=1, norm=true.

using EnergyFlow

# Wrapped azimuthal separation in [0, π]
wrap_dphi(a, b) = (d = abs(a - b); π - abs(mod(d, 2π) - π))

"""
    ring_reference(n) -> n×2 Matrix

Quasi-uniform ring event: `n` particles of equal weight at
φ_j = 2π(j - 1/2)/n. Columns are [weight, φ] (weights are normalized away
by `norm=true`).
"""
ring_reference(n::Int) = hcat(fill(1.0 / n, n), [2π * (j - 0.5) / n for j in 1:n])

"""
    cylinder_reference(nphi, ymax) -> m×3 Matrix

Quasi-uniform cylinder event: `nphi` slices in φ and
`floor(ymax * nphi / π)` slices in y over |y| ≤ ymax (equal grid spacing in
both directions, matching `cylGen.cylinderGen`). Columns are [weight, y, φ].
"""
function cylinder_reference(nphi::Int, ymax::Real)
    ny = floor(Int, ymax * nphi / π)
    phis = [2π * (j - 0.5) / nphi for j in 1:nphi]
    ys = [-ymax + 2ymax * (i - 0.5) / ny for i in 1:ny]
    pts = [(y, phi) for phi in phis for y in ys]
    n = length(pts)
    hcat(fill(1.0 / n, n), first.(pts), last.(pts))
end

"""
    ring_cos_metric()

Ground distance on the ring, (π/(π-2))·(1 - cos Δφ). Events are M×2
matrices [pT, φ]. Matches `emdVar._cdist_phicos`.
"""
ring_cos_metric() = CustomMetric((p, q) -> (π / (π - 2)) * (1 - cos(wrap_dphi(p[1], q[1]))))

"""
    ring_phi_metric()

Ground distance on the ring, (4/π)·Δφ. Matches `emdVar._cdist_phi`.
"""
ring_phi_metric() = CustomMetric((p, q) -> (4 / π) * wrap_dphi(p[1], q[1]))

"""
    cylinder_metric(ymax)

β=2 ground distance on the cylinder, 12/(π² + 16·ymax²)·(Δy² + Δφ²).
Events are M×3 matrices [pT, y, φ]. Matches `emdVar._cdist_phi_y`.
"""
function cylinder_metric(ymax::Real)
    c = 12.0 / (π^2 + 16.0 * ymax^2)
    CustomMetric((p, q) -> begin
        dy = p[1] - q[1]
        dphi = wrap_dphi(p[2], q[2])
        c * (dy * dy + dphi * dphi)
    end)
end

# Event views for the two geometries
ring_event(ev) = ev[:, [1, 3]]   # [pT, φ]

"""
    select_events(events, ymax; min_particles=2) -> Vector{Matrix}

Acceptance preselection for isotropy: keep particles with |y| ≤ `ymax` and
drop events with fewer than `min_particles` accepted particles (the EMD to a
reference is ill-defined for empty events). Apply the same selection in any
implementation being compared against.
"""
select_events(events, ymax::Real; min_particles::Int=2) =
    [ev[abs.(ev[:, 2]) .<= ymax, :] for ev in events
     if count(abs.(ev[:, 2]) .<= ymax) >= min_particles]

"""
    event_isotropy(event, ref, metric; backend=:ns64) -> Float64

EMD between `event` and the quasi-uniform reference `ref` with the given
isotropy ground `metric` (normalization baked in ⇒ R=1, beta=1, norm=true).
"""
event_isotropy(event, ref, metric; backend::Symbol=:ns64) =
    emd(event, ref; R=1.0, beta=1.0, norm=true, metric=metric, backend=backend)

# ─────────────────────────────────────────────────────────────────────
# Spherical geometry — the e⁺e⁻ / lepton-collider version
# ─────────────────────────────────────────────────────────────────────
# The reference is an evenly-tiled unit sphere (HEALPix), events are energy-
# weighted with full 3-momentum directions, and the ground distance is a
# function of the opening angle between directions. Matches spherGen.py and
# emdVar._cdist_cos / _cdist_sphere.

"""
    healpix_pix2vec_ring(nside) -> Vector{NTuple{3,Float64}}

Unit 3-vectors at the centers of the `12·nside²` HEALPix pixels (RING scheme).
Reproduces `astropy_healpix.healpy.pix2vec` (agrees to ~1e-11), which is what
`spherGen.sphericalGen` uses. `nside` must be a power of two.
"""
function healpix_pix2vec_ring(nside::Int)
    npix = 12 * nside^2
    ncap = 2 * nside * (nside - 1)
    vecs = Vector{NTuple{3,Float64}}(undef, npix)
    for p in 0:npix-1
        if p < ncap                                   # north polar cap
            ph = (p + 1) / 2
            i = floor(Int, sqrt(ph - sqrt(floor(ph)))) + 1
            j = p + 1 - 2 * i * (i - 1)
            z = 1.0 - i^2 / (3.0 * nside^2)
            phi = (π / (2i)) * (j - 0.5)
        elseif p < npix - ncap                        # equatorial belt
            pp = p - ncap
            i = pp ÷ (4 * nside) + nside
            j = (pp % (4 * nside)) + 1
            s = (i - nside + 1) % 2
            z = 4.0 / 3.0 - 2.0 * i / (3.0 * nside)
            phi = (π / (2 * nside)) * (j - s / 2.0)
        else                                          # south polar cap
            pp = npix - p
            ph = pp / 2
            i = floor(Int, sqrt(ph - sqrt(floor(ph)))) + 1
            j = 4 * i + 1 - (pp - 2 * i * (i - 1))
            z = -1.0 + i^2 / (3.0 * nside^2)
            phi = (π / (2i)) * (j - 0.5)
        end
        sth = sqrt(1 - z * z)
        vecs[p+1] = (sth * cos(phi), sth * sin(phi), z)
    end
    return vecs
end

"""
    sphere_reference(nVal) -> m×4 Matrix

Quasi-uniform spherical event: equal-weight points at the `12·(2^nVal)²`
HEALPix pixel centers. Columns are [weight, x, y, z] (unit directions).
`nVal = 2` gives 192 points, `nVal = 1` gives 48.
"""
function sphere_reference(nVal::Int)
    vecs = healpix_pix2vec_ring(2^nVal)
    n = length(vecs)
    hcat(fill(1.0 / n, n), getindex.(vecs, 1), getindex.(vecs, 2), getindex.(vecs, 3))
end

# cos of the opening angle between two direction vectors (clamped for safety)
_cosang(p, q) = clamp((p[1]*q[1] + p[2]*q[2] + p[3]*q[3]) /
                      (sqrt(p[1]^2 + p[2]^2 + p[3]^2) * sqrt(q[1]^2 + q[2]^2 + q[3]^2)),
                      -1.0, 1.0)

"""
    sphere_cos_metric()

Ground distance on the sphere, 2·(1 − cos θ) between 3-momentum directions.
Events are M×4 matrices [E, px, py, pz]. Matches `emdVar._cdist_cos`.
"""
sphere_cos_metric() = CustomMetric((p, q) -> 2.0 * (1.0 - _cosang(p, q)))

"""
    sphere_angular_metric()

Angular ground distance on the sphere, arccos(cos θ) ∈ [0, π]. Matches the
`beta`-generic `emdVar._cdist_sphere` at `beta → 0`⁺ up to the arccos form
(here the plain opening angle). Events are M×4 matrices [E, px, py, pz].
"""
sphere_angular_metric() = CustomMetric((p, q) -> acos(_cosang(p, q)))

# Event view: keep particles with non-zero momentum, [E, px, py, pz]
sphere_event(ev) = ev[vec(sum(abs2, @view(ev[:, 2:4]); dims=2)) .> 0, :]

"""
    select_sphere_events(events; min_particles=2) -> Vector{Matrix}

Preselection for spherical isotropy: drop zero-momentum particles and any
event with fewer than `min_particles` remaining.
"""
select_sphere_events(events; min_particles::Int=2) =
    [se for ev in events for se in (sphere_event(ev),) if size(se, 1) >= min_particles]

"""
    load_hepmc3_momenta(filepath; maxevents=-1, status=1) -> Vector{Matrix{Float64}}

Load events from a HepMC3 ASCII file as `M×4` matrices with columns
[E, px, py, pz] — the input format for spherical isotropy. Keeps final-state
particles with the given `status` code.
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
