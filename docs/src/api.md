# API Reference

## Package

```@docs
EnergyFlow
```

## High-level API

The recommended entry points. These dispatch to the currently active backend
(see [`set_backend`](@ref)).

```@docs
emd
emd!
emds
emds!
```

## Backend selection

```@docs
set_backend
get_backend
```

## Ground metrics

```@docs
GroundMetric
EuclideanMetric
SquaredEuclideanMetric
EtaPhiMetric
PrecomputedMetric
CustomMetric
```

## Workspaces

```@docs
EMDWorkspace
SinkhornWorkspace
```

## Event I/O

```@docs
EnergyFlow.HepMC3
EnergyFlow.load_hepmc3_events
load_hepmc3_momenta
```

## Event Isotropy

```@docs
ring_reference
cylinder_reference
sphere_reference
healpix_pix2vec_ring
ring_cos_metric
ring_phi_metric
cylinder_metric
sphere_cos_metric
sphere_angular_metric
select_rapidity
select_sphere_events
event_isotropy
```

## Backend-specific functions

Direct entry points for each solver backend, bypassing backend dispatch.

### Network simplex, Float64 (`:ns64`)

```@docs
emd_ns64
emd_ns64!
emds_ns64
emds_ns64!
```

### OT-style arc mixing, Float64 (`:ot64`)

```@docs
emd_ot64
emd_ot64!
emds_ot64
emds_ot64!
```

### Network simplex, Float32 (`:ns32`)

```@docs
emd_ns32
emd_ns32!
emds_ns32
emds_ns32!
```

### OT-style arc mixing, Float32 (`:ot32`)

```@docs
emd_ot32
emd_ot32!
emds_ot32
emds_ot32!
```

### Sinkhorn (`:sinkhorn`)

```@docs
emd_sinkhorn
emd_sinkhorn!
emds_sinkhorn
```

## Low-level solver

The network simplex solver underlying the exact backends. Most users should
not need these directly.

```@docs
NetworkSimplexSolver
network_simplex!
```
