# Event Isotropy Example — EnergyFlow.jl

Files in this folder:

- [example_isotropy.jl](example_isotropy.jl) — runnable Julia script

Run the script:

```bash
cd example/isotropy
julia --project=. example_isotropy.jl
```

## What is event isotropy?

Event isotropy
([Cesarotti & Thaler, arXiv:2004.06125](https://arxiv.org/abs/2004.06125))
is an event-shape observable defined as the EMD between an event and a
*quasi-uniform reference event*:

```
I(E) = EMD(E, U)
```

where `U` is a uniform radiation pattern and each event's energy or `pT`
weights are normalized to sum to 1. The reference geometry depends on the
collider:

- **hadron colliders** — a ring of `N` points in azimuth `φ`, or a cylinder
  grid in `(y, φ)`
- **lepton colliders** — a uniform sphere in 3-momentum direction (`I_sph`)

In the continuum definitions, `I → 0` for perfectly isotropic events and
`I → 1` for pencil-like (for example, dijet) configurations. Finite reference
grids and orientation choices can shift these endpoints slightly. Compared to
traditional event shapes such as thrust or sphericity, isotropy provides more
resolution in the nearly isotropic regime.

ATLAS measured the ring and cylinder observables in multijet events
([arXiv:2305.16930](https://arxiv.org/abs/2305.16930)), reporting `IRing2`,
`1 − IRing128`, and `1 − ICyl16`.

## Definitions

The conventions follow the reference POT implementation
([caricesarotti/event_isotropy](https://github.com/caricesarotti/event_isotropy)).
The example uses EnergyFlow.jl's exported geometry and metric helpers:

```julia
ring_reference(128)
cylinder_reference(16, 4.0)
sphere_reference(2)
ring_cos_metric()
cylinder_metric(4.0)
sphere_cos_metric()
event_isotropy(event, ref, metric)
event_isotropy(event; geometry=:ring, n=128)
```

Each ground distance includes its normalization, so every EMD is evaluated
with `R=1`, `beta=1`, and `norm=true`.

Ring events are `M×2` matrices `[pT, φ]`; cylinder events are `M×3` matrices
`[pT, y, φ]`; sphere events are `M×4` matrices `[E, px, py, pz]`.

## Spherical isotropy (lepton colliders)

For e⁺e⁻ events the reference is a uniform sphere of radiation, built from the
centers of the `12·nside²` HEALPix pixels (`nside = 2^nval`) — the same tiling
used by the reference
[`event_isotropy`](https://github.com/caricesarotti/event_isotropy) package via
`astropy_healpix`. EnergyFlow.jl provides `healpix_pix2vec_ring` and
`sphere_reference`, so no extra HEALPix dependency is needed:

```julia
sph192 = sphere_reference(2)          # 192 points (nval = 2; nval = 1 → 48)
msph   = sphere_cos_metric()          # 2·(1 − cos θ) on 3-momentum directions
I_sph  = emd(event, sph192;
             R=1.0, beta=1.0, norm=true, metric=msph)
```

The example checks the range on toy events: a back-to-back dijet gives
`I_sph ≈ 1`, a deterministic isotropic (Fibonacci-sphere) event gives
`I_sph ≈ 0.01`, and the sphere reference itself gives exactly `0`. The forward
`pp` sample, read as a sphere, sits near `1`, as expected for collimated forward
radiation; genuine e⁺e⁻ data would populate the low-`I_sph` region.

## Sanity checks

A back-to-back dijet toy event gives `IRing128 ≈ 1` (maximally anisotropic); an
event that is the 128-point ring gives exactly `0`.

## Acceptance selection

Events are preselected to particles with `|y| ≤ ymax` (default `4.0`), and
events with fewer than two accepted particles are dropped. This acceptance is
part of the example's observable definition; apply the same selection in any
implementation you compare against.

## IRing2 (ATLAS convention)

The 2-point ring gives a thrust-like observable. Following ATLAS, the ground
distance is normalized by `1/(1 − 1/√3)` instead of `π/(π−2)`. Because a
2-point ring is not rotationally symmetric, the example minimizes the EMD over
the ring orientation with a one-dimensional scan.

## Tight loops

For many events against a fixed reference, construct an `EMDWorkspace` (the
metric is stored in the workspace) and call `emd!`:

```julia
n = maximum(size(ring_event(ev), 1) for ev in events)
ws = EMDWorkspace(n, size(ring128, 1);
                  beta=1.0, R=1.0, norm=true, metric=ring_cos_metric())
vals = [emd!(ws, ring_event(ev), ring128) for ev in events]
```

## Benchmarks

`benchmark/isotropy_benchmark.jl` times these observables over 100 events per
exact solver backend, and `benchmark/isotropy_benchmark_python.py` runs the
matched computation through POT's `ot.lp.emd2`. The comparison script reports
both timing and the difference in mean isotropy. The AirspeedVelocity suite
exercises the same package helpers in `benchmark/benchmarks.jl`, group
`"isotropy"`.
