# Distance computation functions

"""
    compute_distance_matrix(coords1::Matrix{Float64}, coords2::Matrix{Float64}, params::EMDParameters)

Compute pairwise distance matrix between two sets of coordinates.
"""
function compute_distance_matrix(coords1::Matrix{Float64}, coords2::Matrix{Float64}, 
                               params::EMDParameters)
    
    n1, n2 = size(coords1, 1), size(coords2, 1)
    dist_matrix = zeros(n1, n2)
    
    if params.measure == "euclidean"
        if params.periodic_phi && size(coords1, 2) >= 2
            # Handle periodic phi (assuming phi is last column)
            phi_col = size(coords1, 2)
            
            for i in 1:n1
                for j in 1:n2
                    # Periodic distance in phi
                    d_phi = π - abs(π - abs(coords1[i, phi_col] - coords2[j, phi_col]))
                    
                    # Euclidean distance in other coordinates
                    d_others_sq = 0.0
                    for k in 1:(phi_col-1)
                        diff = coords1[i, k] - coords2[j, k]
                        d_others_sq += diff^2
                    end
                    
                    dist_matrix[i, j] = sqrt(d_others_sq + d_phi^2)
                end
            end
        else
            # Standard Euclidean distance
            for i in 1:n1
                for j in 1:n2
                    dist_sq = 0.0
                    for k in 1:size(coords1, 2)
                        diff = coords1[i, k] - coords2[j, k]
                        dist_sq += diff^2
                    end
                    dist_matrix[i, j] = sqrt(dist_sq)
                end
            end
        end
    else
        # Spherical measure (opening angles)
        for i in 1:n1
            for j in 1:n2
                cos_angle = clamp(dot(coords1[i, :], coords2[j, :]), -1.0, 1.0)
                dist_matrix[i, j] = acos(cos_angle)
            end
        end
    end
    
    # Scale by R parameter
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