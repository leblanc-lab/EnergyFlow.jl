# Distance computation functions with SIMD optimization

using LoopVectorization
using StaticArrays

"""
    euclidean_distance_2d(x1::Float64, y1::Float64, x2::Float64, y2::Float64)

Specialized 2D Euclidean distance with SIMD optimization.
"""
@inline @fastmath function euclidean_distance_2d(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
    dx = x1 - x2
    dy = y1 - y2
    return sqrt(muladd(dx, dx, dy * dy))
end

"""
    euclidean_distance_3d(x1::Float64, y1::Float64, z1::Float64, x2::Float64, y2::Float64, z2::Float64)

Specialized 3D Euclidean distance with SIMD optimization.
"""
@inline @fastmath function euclidean_distance_3d(x1::Float64, y1::Float64, z1::Float64,
                                                 x2::Float64, y2::Float64, z2::Float64)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return sqrt(muladd(dx, dx, muladd(dy, dy, dz * dz)))
end

"""
    periodic_phi_distance(phi1::Float64, phi2::Float64)

Compute periodic distance in phi coordinate.
"""
@inline @fastmath function periodic_phi_distance(phi1::Float64, phi2::Float64)
    d_phi = abs(phi1 - phi2)
    return π - abs(π - d_phi)
end

"""
    compute_distance_matrix_2d_specialized(coords1, coords2, R::Float64)

Specialized 2D distance matrix computation with @turbo optimization.
"""
@inline function compute_distance_matrix_2d_specialized(coords1::AbstractMatrix{Float64},
                                                       coords2::AbstractMatrix{Float64},
                                                       R::Float64)
    n1, n2 = size(coords1, 1), size(coords2, 1)
    dist_matrix = Matrix{Float64}(undef, n1, n2)

    # Use @turbo for vectorized computation
    @turbo for i in 1:n1
        for j in 1:n2
            dx = coords1[i, 1] - coords2[j, 1]
            dy = coords1[i, 2] - coords2[j, 2]
            dist_matrix[i, j] = sqrt(muladd(dx, dx, dy * dy)) / R
        end
    end

    return dist_matrix
end

"""
    compute_distance_matrix_3d_specialized(coords1, coords2, R::Float64)

Specialized 3D distance matrix computation with @turbo optimization.
"""
@inline function compute_distance_matrix_3d_specialized(coords1::AbstractMatrix{Float64},
                                                       coords2::AbstractMatrix{Float64},
                                                       R::Float64)
    n1, n2 = size(coords1, 1), size(coords2, 1)
    dist_matrix = Matrix{Float64}(undef, n1, n2)

    # Use @turbo for vectorized computation
    @turbo for i in 1:n1
        for j in 1:n2
            dx = coords1[i, 1] - coords2[j, 1]
            dy = coords1[i, 2] - coords2[j, 2]
            dz = coords1[i, 3] - coords2[j, 3]
            dist_matrix[i, j] = sqrt(muladd(dx, dx, muladd(dy, dy, dz * dz))) / R
        end
    end

    return dist_matrix
end

"""
    compute_distance_matrix(coords1::AbstractMatrix{Float64}, coords2::AbstractMatrix{Float64}, params::EMDParameters)

Compute pairwise distance matrix between two sets of coordinates.
Optimized with SIMD vectorization for improved performance.
"""
function compute_distance_matrix(coords1::AbstractMatrix{Float64}, coords2::AbstractMatrix{Float64}, 
                               params::EMDParameters)
    
    n1, n2 = size(coords1, 1), size(coords2, 1)
    dist_matrix = zeros(n1, n2)
    
    if params.measure == "euclidean"
        # Use specialized functions for common cases
        if size(coords1, 2) == 2 && !params.periodic_phi
            return compute_distance_matrix_2d_specialized(coords1, coords2, params.R)
        elseif size(coords1, 2) == 3 && !params.periodic_phi
            return compute_distance_matrix_3d_specialized(coords1, coords2, params.R)
        elseif params.periodic_phi && size(coords1, 2) >= 2
            # Handle periodic phi with @turbo optimization
            phi_col = size(coords1, 2)

            @turbo for i in 1:n1
                for j in 1:n2
                    # Periodic distance in phi
                    d_phi = periodic_phi_distance(coords1[i, phi_col], coords2[j, phi_col])

                    # Euclidean distance in other coordinates
                    d_others_sq = 0.0
                    for k in 1:(phi_col-1)
                        diff = coords1[i, k] - coords2[j, k]
                        d_others_sq = muladd(diff, diff, d_others_sq)
                    end

                    dist_matrix[i, j] = sqrt(muladd(d_phi, d_phi, d_others_sq))
                end
            end
        else
            # Standard Euclidean distance with @turbo optimization
            @turbo for i in 1:n1
                for j in 1:n2
                    dist_sq = 0.0
                    for k in 1:size(coords1, 2)
                        diff = coords1[i, k] - coords2[j, k]
                        dist_sq = muladd(diff, diff, dist_sq)
                    end
                    dist_matrix[i, j] = sqrt(dist_sq)
                end
            end
        end
    else
        # Spherical distance with @turbo optimization
        @turbo for i in 1:n1
            for j in 1:n2
                dot_product = 0.0
                for k in 1:size(coords1, 2)
                    dot_product = muladd(coords1[i, k], coords2[j, k], dot_product)
                end
                cos_angle = clamp(dot_product, -1.0, 1.0)
                dist_matrix[i, j] = acos(cos_angle)
            end
        end
    end
    
    # Scale by R
    dist_matrix = dist_matrix / params.R
    
    return dist_matrix
end

"""
    pairwise_distances(coords::Matrix{Float64}, params::EMDParameters)

Compute pairwise distances within a single set of coordinates.
"""
function pairwise_distances(coords::Matrix{Float64}, params::EMDParameters)
    return compute_distance_matrix(coords, coords, params)
end