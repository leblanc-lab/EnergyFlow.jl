# Utility functions for EnergyFlow

"""
    cartesian_to_hadronic(event::Matrix{Float64})

Convert cartesian coordinates (E, px, py, pz) to hadronic (pT, y, phi).
"""
function cartesian_to_hadronic(event::Matrix{Float64})
    n_particles = size(event, 1)
    hadronic = zeros(n_particles, 3)
    
    E = event[:, 1]
    px = event[:, 2]
    py = event[:, 3]
    pz = event[:, 4]
    
    # pT = sqrt(px^2 + py^2)
    hadronic[:, 1] = sqrt.(px.^2 + py.^2)
    
    # y = 1/2 * log((E + pz)/(E - pz))
    hadronic[:, 2] = 0.5 * log.((E + pz) ./ (E - pz))
    
    # phi = atan2(py, px)
    hadronic[:, 3] = atan.(py, px)
    
    return hadronic
end

"""
    apply_kinematic_cuts(event::Matrix{Float64}; pT_min=0.0, y_max=Inf, 
                        coords="hadronic")

Apply kinematic cuts to particle events.

# Arguments
- `event`: Event matrix
- `pT_min`: Minimum transverse momentum cut
- `y_max`: Maximum rapidity cut
- `coords`: Coordinate system ("hadronic" or "cartesian")

# Returns
- `Matrix{Float64}`: Event with cuts applied
"""
function apply_kinematic_cuts(event::Matrix{Float64}; 
                            pT_min=0.0, y_max=Inf, 
                            coords="hadronic")
    
    if coords == "cartesian"
        # Convert to hadronic for cuts
        hadronic = cartesian_to_hadronic(event)
        pT = hadronic[:, 1]
        y = hadronic[:, 2]
    else
        pT = event[:, 1]
        y = event[:, 2]
    end
    
    # Apply cuts
    mask = (pT .>= pT_min) .& (abs.(y) .<= y_max)
    
    return event[mask, :]
end

"""
    mask_particles_outside_radius(event::Matrix{Float64}, R::Float64; 
                                 coords="hadronic", measure="euclidean")

Mask particles outside a given radius R from the origin.
"""
function mask_particles_outside_radius(event::Matrix{Float64}, R::Float64; 
                                     coords="hadronic", measure="euclidean")
    
    if measure == "euclidean"
        if coords == "hadronic"
            # Use y-phi distance
            y = event[:, 2]
            phi = event[:, 3]
            distances = sqrt.(y.^2 + phi.^2)
        else
            # Use spatial distance
            coords_spatial = event[:, 2:end]
            distances = [norm(coords_spatial[i, :]) for i in 1:size(coords_spatial, 1)]
        end
    else
        # For spherical measure, R represents angular distance
        # This requires more complex handling
        distances = zeros(size(event, 1))
    end
    
    mask = distances .<= R
    return event[mask, :]
end

"""
    total_momentum(event::Matrix{Float64}; coords="hadronic")

Calculate total four-momentum of an event.

# Returns
- `Vector{Float64}`: [E, px, py, pz] for total momentum
"""
function total_momentum(event::Matrix{Float64}; coords="hadronic")
    if coords == "hadronic"
        # Convert to cartesian first
        pT = event[:, 1]
        y = event[:, 2]
        phi = event[:, 3]
        
        E = pT .* cosh.(y)
        px = pT .* cos.(phi)
        py = pT .* sin.(phi)
        pz = pT .* sinh.(y)
        
        return [sum(E), sum(px), sum(py), sum(pz)]
    else
        # Already in cartesian
        return vec(sum(event, dims=1))
    end
end

"""
    center_of_energy(event::Matrix{Float64}; coords="hadronic")

Calculate the energy-weighted centroid of an event.

# Returns
- `Vector{Float64}`: Centroid coordinates in the same system as input
"""
function center_of_energy(event::Matrix{Float64}; coords="hadronic")
    if coords == "hadronic"
        pT = event[:, 1]
        y = event[:, 2]
        phi = event[:, 3]
        
        # Weight by pT
        total_pT = sum(pT)
        if total_pT > 0
            y_avg = sum(pT .* y) / total_pT
            
            # For phi, handle circular average
            phi_x = sum(pT .* cos.(phi)) / total_pT
            phi_y = sum(pT .* sin.(phi)) / total_pT
            phi_avg = atan(phi_y, phi_x)
            
            return [total_pT, y_avg, phi_avg]
        else
            return [0.0, 0.0, 0.0]
        end
    else
        # Cartesian coordinates
        E = event[:, 1]
        total_E = sum(E)
        
        if total_E > 0
            weighted_coords = sum(E .* event, dims=1) / total_E
            return vec(weighted_coords)
        else
            return zeros(size(event, 2))
        end
    end
end