# Event Isotropy Example — EnergyFlow.jl

Files in this folder:

- [example_isotropy.jl](example_isotropy.jl) — runnable Julia script

Run the script:
```bash
cd example/isotropy
julia --project=. example_isotropy.jl
```

## What is event isotropy?

Event isotropy ([Cesarotti & Thaler, arXiv:2004.06125](https://arxiv.org/abs/2004.06125)) is an event-shape observable defined as the EMD between an event and a *quasi-uniform reference event*:

```
I(E) = EMD(E, U)
```

where `U` is a uniform radiation pattern and each event's energy/pT weights are normalized to sum to 1. The reference geometry depends on the collider:

- **hadron colliders** — a ring of `N` points in azimuth `φ`, or a cylinder grid in `(y, φ)`;
- **lepton colliders** — a uniform sphere in 3-momentum direction (`I_sph`).

`I → 0` for perfectly isotropic events and `I → 1` for pencil-like (e.g. dijet) configurations. Compared to traditional event shapes like thrust or sphericity, it has much greater dynamic range in the nearly-isotropic regime.

ATLAS measured the ring/cylinder observables in multijet events ([arXiv:2305.16930](https://arxiv.org/abs/2305.16930)), reporting `IRing2`, `1 − IRing128` and `1 − ICyl16`.

## Definitions

The conventions follow the reference POT implementation ([caricesarotti/event_isotropy](https://github.com/caricesarotti/event_isotropy)). Each ground distance bakes its normalization into the metric, so every EMD is evaluated with `R=1`, `beta=1`, `norm=true`.

**Reference events**

| geometry | construction | weights |
|----------|--------------|---------|
| ring `N` | `φ_j = 2π(j − ½)/N` | equal |
| cylinder `n_φ × n_y` | `n_φ` slices in φ × `floor(ymax·n_φ/π)` slices in `y`, over `\|y\| ≤ ymax` | equal |
| sphere `nVal` | `12·(2^nVal)²` HEALPix pixel centers (RING scheme) | equal |

**Ground distances**

| observable | distance |
|------------|----------|
| ring ("cos" measure) | `(π/(π−2)) · (1 − cos Δφ)` |
| cylinder (β=2 measure) | `12/(π² + 16·ymax²) · (Δy² + Δφ²)` |
| sphere ("cos" measure) | `2 · (1 − cos θ)` on 3-momentum directions |

with `Δφ` wrapped to `[0, π]` and `θ` the opening angle. These map directly onto `CustomMetric`:

```julia
wrap_dphi(a, b) = (d = abs(a - b); π - abs(mod(d, 2π) - π))

ring_cos_metric() =
    CustomMetric((p, q) -> (π / (π - 2)) * (1 - cos(wrap_dphi(p[1], q[1]))))

event_isotropy(event, ref, metric) =
    emd(event, ref; R=1.0, beta=1.0, norm=true, metric=metric)
```

Ring events are `M×2` matrices `[pT, φ]`; cylinder events are the usual `M×3` `[pT, y, φ]`; sphere events are `M×4` matrices `[E, px, py, pz]`.

## Spherical isotropy (lepton colliders)

For e⁺e⁻ events the reference is a uniform sphere of radiation, built from the centers of the `12·nside²` HEALPix pixels (`nside = 2^nVal`) — the same tiling used by the reference [`event_isotropy`](https://github.com/caricesarotti/event_isotropy) package via `astropy_healpix`. The example reproduces `astropy_healpix.pix2vec` in pure Julia (agrees to ~1e-11), so no HEALPix dependency is needed:

```julia
sph192 = sphere_reference(2)          # 192-point sphere (nVal = 2; nVal = 1 → 48)
msph   = sphere_cos_metric()          # 2·(1 − cos θ) on 3-momentum directions
I_sph  = emd(event, sph192; R=1.0, beta=1.0, norm=true, metric=msph)
```

The example checks the full dynamic range on toy events: a back-to-back dijet gives `I_sph ≈ 1`, a deterministic isotropic (Fibonacci-sphere) event gives `I_sph ≈ 0.01`, and the sphere reference itself gives exactly `0`. The forward pp sample, read as a sphere, sits near `1` — as expected for collimated forward radiation; genuine e⁺e⁻ data would populate the low-`I_sph` region.

## Sanity checks

A back-to-back dijet toy event gives `IRing128 ≈ 1` (maximally anisotropic); an event that *is* the 128-point ring gives exactly `0`.

## Acceptance selection

Events are preselected to particles with `|y| ≤ ymax` (default `4.0`), and events with fewer than 2 accepted particles are dropped — the EMD to a reference is ill-defined for empty events. Apply the same selection in any implementation you compare against.

## IRing2 (ATLAS convention)

The 2-point ring gives a thrust-like observable. Following ATLAS, the ground distance is normalized by `1/(1 − 1/√3)` instead of `π/(π−2)`, and because a 2-point ring is not rotationally symmetric, the EMD is minimized over the ring orientation (a 1-D scan in the example).

## Tight loops

For many events against a fixed reference, pre-allocate an `EMDWorkspace` (the metric is stored in the workspace) and call `emd!`:

```julia
n = maximum(size(ring_event(ev), 1) for ev in events)
ws = EMDWorkspace(n, size(ring128, 1); beta=1.0, R=1.0, norm=true, metric=ring_cos_metric())
vals = [emd!(ws, ring_event(ev), ring128) for ev in events]
```

## Benchmarks

`benchmark/isotropy_benchmark.jl` times these observables over 100 events per solver backend, and `benchmark/isotropy_benchmark_python.py` runs the identical computation through POT's `ot.lp.emd2` — the mean isotropy values printed by both scripts agree to ≥ 6 decimal places, and the same definitions are exercised by the CI benchmark suite (`benchmark/benchmarks.jl`, group `"isotropy"`).
