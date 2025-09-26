export AbstractPairwiseDistance, DefaultPairwiseDistance, compute_distance_matrix!,
       compute_distance, compute_distance_2d, AngularDistance, compute_angular_distance_2d,
       compute_distance_static

# Abstract base type for pairwise distance calculations
abstract type AbstractPairwiseDistance{V<:AbstractFloat} end

# Default implementation using Euclidean distance
mutable struct DefaultPairwiseDistance{V<:AbstractFloat} <: AbstractPairwiseDistance{V}
    R::V
    R2::V
    beta::V
    halfbeta::V
    
    function DefaultPairwiseDistance{V}(R::V=one(V), beta::V=one(V)) where V
        if R <= 0
            throw(ArgumentError("R must be positive"))
        end
        if beta < 0
            throw(ArgumentError("beta must be non-negative"))
        end
        new{V}(R, R*R, beta, beta/2)
    end
end

DefaultPairwiseDistance(R::Real=1.0, beta::Real=1.0) = 
    DefaultPairwiseDistance{Float64}(Float64(R), Float64(beta))

# Compute plain squared distance between two particles (SIMD optimized with LoopVectorization)
function plain_distance(p0::AbstractVector{V}, p1::AbstractVector{V}) where V
    dist = zero(V)
    @turbo for i in eachindex(p0, p1)
        diff = p0[i] - p1[i]
        dist += diff * diff
    end
    return dist
end

# Specialized version for StaticArrays - extremely fast for small dimensions
@inline function plain_distance(p0::SVector{N,V}, p1::SVector{N,V}) where {N,V}
    diff = p0 - p1  # SIMD optimized by StaticArrays
    return dot(diff, diff)  # Uses optimized dot product
end

# Generated function for specialized distance computation based on beta value
@generated function compute_distance(pd::DefaultPairwiseDistance{V}, p0::AbstractVector, p1::AbstractVector, ::Val{beta}) where {V, beta}
    if beta == 1.0
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath sqrt(plain_dist) / pd.R
        end
    elseif beta == 2.0
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath plain_dist / pd.R2
        end
    else
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Fallback for runtime beta values
@inline function compute_distance(pd::DefaultPairwiseDistance{V}, p0::AbstractVector, p1::AbstractVector) where V
    plain_dist = plain_distance(p0, p1)
    return compute_distance_from_plain(pd, plain_dist)
end

# Helper function to compute distance from plain distance
@inline function compute_distance_from_plain(pd::DefaultPairwiseDistance{V}, plain_dist::V) where V
    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Generated function for specialized 2D distance computation
@generated function compute_distance_2d(pd::DefaultPairwiseDistance{V}, p0_1::V, p0_2::V, p1_1::V, p1_2::V, ::Val{beta}) where {V, beta}
    if beta == 1.0
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                sqrt(plain_dist) / pd.R
            end
        end
    elseif beta == 2.0
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                plain_dist / pd.R2
            end
        end
    else
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                (plain_dist / pd.R2)^pd.halfbeta
            end
        end
    end
end

# Fallback for runtime beta values
@inline function compute_distance_2d(pd::DefaultPairwiseDistance{V}, p0_1::V, p0_2::V, p1_1::V, p1_2::V) where V
    @fastmath begin
        diff1 = p0_1 - p1_1
        diff2 = p0_2 - p1_2
        plain_dist = muladd(diff1, diff1, diff2 * diff2)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Fill distance matrix for two particle collections
function compute_distance_matrix!(pd::AbstractPairwiseDistance{V},
                                  particles0::Matrix{V},
                                  particles1::Matrix{V},
                                  dists::Vector{V},
                                  extra_particle::Symbol=:neither,
                                  extra_cost::V=one(V)) where V

    n0_orig = size(particles0, 2)
    n1_orig = size(particles1, 2)

    # Determine actual dimensions including dummy
    n0 = n0_orig + (extra_particle == :zero ? 1 : 0)
    n1 = n1_orig + (extra_particle == :one ? 1 : 0)
    
    # Compute pairwise distances
    k = 1

    if extra_particle == :zero
        # Extra particle added to event 0 (as last particle)
        # Fill matrix row by row: first n0_orig real particles, then dummy

        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            # First n0_orig rows: real particles from event0
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
            end
            # Last row: distances from dummy particle to all event1 particles
            @inbounds for j in 1:n1_orig
                dists[k] = extra_cost
                k += 1
            end
        else
            # First n0_orig rows: real particles from event0
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
            end
            # Last row: distances from dummy particle to all event1 particles
            @inbounds for j in 1:n1_orig
                dists[k] = extra_cost
                k += 1
            end
        end
    elseif extra_particle == :one
        # Extra particle added to event 1 (as last particle)
        # Fill matrix row by row: for each event0 particle, all n1_orig real + 1 dummy

        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            @inbounds for i in 1:n0_orig
                # First n1_orig columns: real particles from event1
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
                # Last column: distance to dummy particle
                dists[k] = extra_cost
                k += 1
            end
        else
            @inbounds for i in 1:n0_orig
                # First n1_orig columns: real particles from event1
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
                # Last column: distance to dummy particle
                dists[k] = extra_cost
                k += 1
            end
        end
    else
        # No extra particle - standard case
        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
            end
        else
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
            end
        end
    end

    return nothing
end

# Specialized distances for physics applications

# Angular distance for particles with (eta, phi) coordinates
struct AngularDistance{V<:AbstractFloat} <: AbstractPairwiseDistance{V}
    R::V
    R2::V
    beta::V
    halfbeta::V
    
    function AngularDistance{V}(R::V=one(V), beta::V=one(V)) where V
        if R <= 0
            throw(ArgumentError("R must be positive"))
        end
        if beta < 0
            throw(ArgumentError("beta must be non-negative"))
        end
        new{V}(R, R*R, beta, beta/2)
    end
end

@inline function compute_distance(pd::AngularDistance{V}, p0::AbstractVector, p1::AbstractVector) where V
    @fastmath @inbounds begin
        # Assuming p0 and p1 have (eta, phi) coordinates
        deta = p0[1] - p1[1]
        dphi = p0[2] - p1[2]

        # Optimized phi wrapping using modulo
        dphi = mod(dphi + π, 2π) - π

        # Compute angular distance
        plain_dist = muladd(deta, deta, dphi * dphi)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Specialized fast version for angular distance with direct coordinate access
@inline function compute_angular_distance_2d(pd::AngularDistance{V}, eta0::V, phi0::V, eta1::V, phi1::V) where V
    @fastmath begin
        deta = eta0 - eta1
        dphi = mod(phi0 - phi1 + π, 2π) - π
        plain_dist = muladd(deta, deta, dphi * dphi)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# High-performance static distance computation for known dimensions
@inline function compute_distance_static(pd::DefaultPairwiseDistance{V}, p0::SVector{2,V}, p1::SVector{2,V}) where V
    diff = p0 - p1
    plain_dist = dot(diff, diff)

    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

@inline function compute_distance_static(pd::DefaultPairwiseDistance{V}, p0::SVector{3,V}, p1::SVector{3,V}) where V
    diff = p0 - p1
    plain_dist = dot(diff, diff)

    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end



# ========== Event Handling ==========



