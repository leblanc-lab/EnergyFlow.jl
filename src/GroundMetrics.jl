#=
    GroundMetrics.jl

    GroundMetrics: Type definitions for ground distance metrics.

    These types are used for dispatch in Distances.jl cost-filling functions.
    Separated into their own file so they can be loaded before EMDSolver.jl
    (which references GroundMetric in the EMDWorkspace struct).
=#

# ─── Ground Metric Types ───

"""
    GroundMetric

Abstract supertype for ground distance metrics used to fill the pairwise
particle cost matrix in an EMD computation.

Concrete subtypes: [`EuclideanMetric`](@ref), [`SquaredEuclideanMetric`](@ref),
[`EtaPhiMetric`](@ref), [`PrecomputedMetric`](@ref), [`CustomMetric`](@ref).

Pass an instance via the `metric` keyword of [`emd`](@ref) / [`emds`](@ref),
or store it in an [`EMDWorkspace`](@ref) for the in-place API.
"""
abstract type GroundMetric end

"""
    EuclideanMetric()

Euclidean ground distance (the default metric).

The cost of moving weight between particles `i` and `j` is

    cost_ij = (√(Σₖ Δxₖ²) / R)^β

computed over all coordinate dimensions of the event.
"""
struct EuclideanMetric <: GroundMetric end

"""
    SquaredEuclideanMetric()

Squared Euclidean ground distance (no square root):

    cost_ij = (Σₖ Δxₖ² / R)^β

Equivalent to `EuclideanMetric` with `beta=2` when `R = 1`. Useful when a
Wasserstein-2-style cost is desired, or to skip the `sqrt` for speed.
"""
struct SquaredEuclideanMetric <: GroundMetric end

"""
    EtaPhiMetric()

Collider-aware (η, φ) ground distance with periodic azimuthal angle:

    cost_ij = (√(Δη² + Δφ_wrap²) / R)^β,   Δφ_wrap = π - |mod(|Δφ|, 2π) - π|

Assumes coordinate dimension 1 is η (rapidity or pseudorapidity) and
dimension 2 is φ (azimuthal angle, radians). This is the appropriate metric
for events in `[pT, η, φ]` format when particles may lie near the φ = ±π
boundary, where a plain `EuclideanMetric` would overestimate distances.
"""
struct EtaPhiMetric <: GroundMetric end

"""
    PrecomputedMetric(costs::AbstractMatrix)

Use a caller-supplied `n0 × n1` cost matrix directly, where `costs[i, j]` is
the cost of transporting weight from particle `i` of the first event to
particle `j` of the second.

Values are used **as-is** — `R` and `beta` are *not* applied. The matrix must
be at least as large as the particle counts of the event pair it is used with,
so this metric is mainly useful for single-pair computations.
"""
struct PrecomputedMetric{M<:AbstractMatrix} <: GroundMetric
    costs::M
end

"""
    CustomMetric(dist_fn)

Compute ground distances with a user-supplied function.

`dist_fn(p, q)` receives views of the coordinate columns of one particle from
each event (length-`d` vectors) and must return a scalar distance. `R` and
`beta` are applied on top of the returned value:

    cost_ij = (dist_fn(pᵢ, qⱼ) / R)^β

# Example
```julia
# Manhattan (L1) ground distance
m = CustomMetric((p, q) -> sum(abs, p .- q))
emd(ev0, ev1; metric=m)
```
"""
struct CustomMetric{F} <: GroundMetric
    dist_fn::F
end
