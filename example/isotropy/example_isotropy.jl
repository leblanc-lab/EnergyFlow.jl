# EnergyFlow.jl — Event Isotropy Example
#
# Event isotropy (Cesarotti & Thaler, arXiv:2004.06125) is an event shape
# defined as the EMD between an event and a quasi-uniform reference event:
#
#     I(E) = EMD(E, U)
#
# where U is a uniform radiation pattern on a ring (in φ) or a cylinder
# (in y–φ) for hadron colliders, or a sphere (in 3-momentum direction) for
# lepton colliders. The event's energy/pT weights are normalized to sum to 1.
# I → 0 for isotropic events and I → 1 for pencil-like (e.g. dijet) events.
#
# ATLAS measured the ring/cylinder observables in multijet events
# (arXiv:2305.16930), reporting IRing2, 1−IRing128 and 1−ICyl16. The spherical
# version is the natural event shape for e⁺e⁻ / lepton colliders.
#
# The ground distances and reference events below follow the reference POT
# implementation (github.com/caricesarotti/event_isotropy); each metric bakes
# its normalization into the ground distance, so all EMDs use R=1, beta=1,
# norm=true.

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using EnergyFlow
using Statistics

wrap_dphi(a, b) = (d = abs(a - b); π - abs(mod(d, 2π) - π))

ring_event(ev) = ev[:, [1, 3]]   # [pT, φ]

ring2_metric() = CustomMetric((p, q) -> (1 - cos(wrap_dphi(p[1], q[1]))) / (1 - 1 / sqrt(3)))

iring2(ev; nscan=64) = begin
    rev, m = ring_event(ev), ring2_metric()
    cost(shift) = event_isotropy(rev, [0.5 shift; 0.5 shift + π], m)
    shifts = range(0, π; length=nscan + 1)[1:end-1]
    costs = cost.(shifts)
    best = shifts[argmin(costs)]
    fine = range(best - π / nscan, best + π / nscan; length=32)
    minimum(cost.(fine))
end

# ── Sanity checks with toy events ──

mring = ring_cos_metric()
ring128 = ring_reference(128)

# A back-to-back dijet event: maximally anisotropic on the ring, I → 1
dijet = [1.0 0.0 0.0;
         1.0 0.0 π]
println("IRing128, dijet toy event:     ", round(event_isotropy(ring_event(dijet), ring128, mring); digits=4))

# A perfectly uniform 128-particle event: I = 0 by construction
println("IRing128, uniform toy event:   ",
        round(event_isotropy(hcat(ring128[:, 1], zeros(128), ring128[:, 2]) |> ring_event,
                             ring128, mring); digits=4))

# ── Real events ──

ymax = 4.0
raw = load_hepmc3_events(joinpath(@__DIR__, "..", "..", "data", "sk_example_PU.hepmc"); maxevents=20)
events = select_events(raw, ymax)
println("\nLoaded $(length(raw)) events, $(length(events)) after |y| ≤ $ymax selection")

# IRing128 — ring reference with 128 points, "1 - cos" ground distance.
iring128 = [event_isotropy(ring_event(ev), ring128, mring) for ev in events]
println("\nIRing128:      mean $(round(mean(iring128); digits=4)), range $(round(minimum(iring128); digits=4)) – $(round(maximum(iring128); digits=4))")

# ATLAS reports 1 − IRing128 (arXiv:2305.16930), so larger = more isotropic
println("1 − IRing128:  mean $(round(1 - mean(iring128); digits=4))")

# ICyl16 — cylinder reference (16 φ-slices, |y| ≤ 4), β=2 ground distance
cyl16 = cylinder_reference(16, ymax)
mcyl = cylinder_metric(ymax)
icyl16 = [event_isotropy(ev, cyl16, mcyl) for ev in events]
println("ICyl16:        mean $(round(mean(icyl16); digits=4))")

# IRing2 — 2-point (back-to-back) ring reference: a thrust-like observable.
# Following ATLAS, the ground distance is normalized by 1/(1 − 1/√3) instead
# of π/(π-2), and the reference orientation is optimized (the 2-point ring is
# not rotationally symmetric, so we minimize the EMD over the ring phase).
vals = [iring2(ev) for ev in events]
println("IRing2:        mean $(round(mean(vals); digits=4))")

# ── Tight loops: reuse a workspace ──
# For many events against a fixed reference, pre-allocate an EMDWorkspace
# (the metric lives in the workspace) and use emd!.
n = maximum(size(ring_event(ev), 1) for ev in events)
ws = EMDWorkspace(n, size(ring128, 1); beta=1.0, R=1.0, norm=true, metric=mring)
iring128_ws = [emd!(ws, ring_event(ev), ring128) for ev in events]
println("\nIRing128 via reusable workspace: mean $(round(mean(iring128_ws); digits=4)) (matches: $(iring128_ws ≈ iring128))")

# ── Spherical isotropy (lepton colliders) ──
# For e⁺e⁻ events the natural reference is a uniform sphere of radiation.
# Events are [E, px, py, pz] (energy weight + 3-momentum); the ground distance
# is 2·(1 − cos θ) on the momentum directions. The reference is an evenly
# tiled unit sphere (HEALPix, 12·nside² points).
println("\n── Spherical isotropy (I_sph) ──")

sph192 = sphere_reference(2)                     # 192-point sphere
msph = sphere_cos_metric()
sphere_isotropy(ev) = emd(ev, sph192; R=1.0, beta=1.0, norm=true, metric=msph)

# Toy events: a back-to-back dijet (pencil-like → I_sph ≈ 1) vs the reference
# sphere itself (perfectly isotropic → I_sph = 0).
dijet = [50.0 0.0 0.0 50.0;
         50.0 0.0 0.0 -50.0]
println("I_sph, dijet toy event:      ", round(sphere_isotropy(dijet); digits=4))
sphere_as_event = hcat(sph192[:, 1], sph192[:, 2], sph192[:, 3], sph192[:, 4])
println("I_sph, uniform sphere event: ", round(sphere_isotropy(sphere_as_event); digits=6))

# A synthetic isotropic e⁺e⁻ event: many particles spread over the full sphere
# (using a Fibonacci sphere so the example is deterministic). Its isotropy sits
# well below a dijet's, showing the observable's dynamic range.
function fibonacci_event(m; energy=1.0)
    golden = π * (3 - sqrt(5))
    rows = Matrix{Float64}(undef, m, 4)
    for k in 1:m
        z = 1 - 2 * (k - 0.5) / m
        r = sqrt(max(0.0, 1 - z^2))
        θ = golden * (k - 1)
        rows[k, :] = [energy, r * cos(θ), r * sin(θ), z]
    end
    rows
end
iso_event = fibonacci_event(200)
println("I_sph, isotropic toy event:  ", round(sphere_isotropy(iso_event); digits=4))

# Real events: the pp sample loaded above is very forward, so read as a sphere
# it looks nearly pencil-like (I_sph ≈ 1). For genuine e⁺e⁻ data you would feed
# final-state [E, px, py, pz] directly. We reconstruct 3-momenta here from the
# [pT, η, φ] events (massless: E = pT·cosh η) to show the mechanics.
function to_momenta(ev)
    pt, η, φ = ev[:, 1], ev[:, 2], ev[:, 3]
    hcat(pt .* cosh.(η), pt .* cos.(φ), pt .* sin.(φ), pt .* sinh.(η))
end
isph = [sphere_isotropy(to_momenta(ev)) for ev in events]
println("I_sph, forward pp events:    mean $(round(mean(isph); digits=4)) (near 1, as expected for forward events)")
