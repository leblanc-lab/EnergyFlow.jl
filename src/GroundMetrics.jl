#=
    GroundMetrics.jl

    GroundMetrics: Type definitions for ground distance metrics.

    These types are used for dispatch in Distances.jl cost-filling functions.
    Separated into their own file so they can be loaded before EMDSolver.jl
    (which references GroundMetric in the EMDWorkspace struct).
=#

# ─── Ground Metric Types ───

abstract type GroundMetric end

"""Default: Euclidean distance. cost = (sqrt(Σ(Δx_k²)) / R)^β"""
struct EuclideanMetric <: GroundMetric end

"""Squared Euclidean: cost = (Σ(Δx_k²))^β / R^β (no sqrt).
Equivalent to EuclideanMetric for β=2 when R is the same."""
struct SquaredEuclideanMetric <: GroundMetric end

"""Eta-Phi with phi periodicity: cost = (sqrt(Δη² + Δφ_wrap²) / R)^β
where Δφ_wrap = π - |mod(|Δφ|, 2π) - π|.
Assumes coords dim 1 = η (rapidity/pseudorapidity), dim 2 = φ (azimuthal angle)."""
struct EtaPhiMetric <: GroundMetric end

"""User provides a precomputed n0×n1 cost matrix directly.
Values are used as-is (R and beta are NOT applied)."""
struct PrecomputedMetric{M<:AbstractMatrix} <: GroundMetric
    costs::M
end

"""User provides a distance function f(view_of_coords0_col_i, view_of_coords1_col_j) -> scalar distance.
R and beta are applied: final_cost = (f(a,b) / R)^β"""
struct CustomMetric{F} <: GroundMetric
    dist_fn::F
end
