# Utils.jl - General utility functions and common parameters

module Utils

using LinearAlgebra
using LoopVectorization

# Common parameters structure used by multiple solvers
struct EMDParameters
    R::Float64
    beta::Float64
    norm::Bool
    measure::String
    periodic_phi::Bool
    gdim::Int
    mask::Bool
    n_iter_max::Int
    epsilon_large_factor::Float64
    epsilon_small_factor::Float64

    function EMDParameters(;
        R::Float64=1.0,
        beta::Float64=1.0,
        norm::Bool=false,
        measure::String="euclidean",
        periodic_phi::Bool=false,
        gdim::Int=2,
        mask::Bool=false,
        n_iter_max::Int=100000,
        epsilon_large_factor::Float64=10000.0,
        epsilon_small_factor::Float64=1.0
    )
        new(R, beta, norm, measure, periodic_phi, gdim, mask, n_iter_max, epsilon_large_factor, epsilon_small_factor)
    end
end

export EMDParameters

# ========== Coordinate Transformation Functions ==========

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

    hadronic[:, 1] = sqrt.(px.^2 + py.^2)
    hadronic[:, 2] = 0.5 * log.((E + pz) ./ (E - pz))
    hadronic[:, 3] = atan.(py, px)

    return hadronic
end

export cartesian_to_hadronic

# ========== Event Processing Functions ==========

"""
    apply_kinematic_cuts(event::Matrix{Float64}; pT_min=0.0, y_max=Inf, coords="hadronic")

Apply kinematic cuts to particle events.
"""
function apply_kinematic_cuts(event::Matrix{Float64};
                            pT_min=0.0, y_max=Inf,
                            coords="hadronic")

    if coords == "cartesian"
        hadronic = cartesian_to_hadronic(event)
        pT = hadronic[:, 1]
        y = hadronic[:, 2]
    else
        pT = event[:, 1]
        y = event[:, 2]
    end

    mask = (pT .>= pT_min) .& (abs.(y) .<= y_max)
    return event[mask, :]
end

"""
    mask_particles_outside_radius(event::Matrix{Float64}, R::Float64; coords="hadronic", measure="euclidean")

Mask particles outside a given radius R from the origin.
"""
function mask_particles_outside_radius(event::Matrix{Float64}, R::Float64;
                                     coords="hadronic", measure="euclidean")

    if measure == "euclidean"
        if coords == "hadronic"
            y = event[:, 2]
            phi = event[:, 3]
            distances = sqrt.(y.^2 + phi.^2)
        else
            coords_spatial = event[:, 2:end]
            distances = [norm(view(coords_spatial, i, :)) for i in axes(coords_spatial, 1)]
        end
    else
        distances = zeros(size(event, 1))
    end

    mask = distances .<= R
    return event[mask, :]
end

export apply_kinematic_cuts, mask_particles_outside_radius

# ========== Physics Quantities ==========

"""
    total_momentum(event::Matrix{Float64}; coords="hadronic")

Calculate total four-momentum of an event.
"""
function total_momentum(event::Matrix{Float64}; coords="hadronic")
    if coords == "hadronic"
        pT = event[:, 1]
        y = event[:, 2]
        phi = event[:, 3]

        E = pT .* cosh.(y)
        px = pT .* cos.(phi)
        py = pT .* sin.(phi)
        pz = pT .* sinh.(y)

        return [sum(E), sum(px), sum(py), sum(pz)]
    else
        return vec(sum(event, dims=1))
    end
end

"""
    center_of_energy(event::Matrix{Float64}; coords="hadronic")

Calculate the energy-weighted centroid of an event.
"""
function center_of_energy(event::Matrix{Float64}; coords="hadronic")
    if coords == "hadronic"
        pT = event[:, 1]
        y = event[:, 2]
        phi = event[:, 3]

        total_pT = sum(pT)
        if total_pT > 0
            y_avg = sum(pT .* y) / total_pT
            phi_x = sum(pT .* cos.(phi)) / total_pT
            phi_y = sum(pT .* sin.(phi)) / total_pT
            phi_avg = atan(phi_y, phi_x)

            return [total_pT, y_avg, phi_avg]
        else
            return [0.0, 0.0, 0.0]
        end
    else
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

export total_momentum, center_of_energy

# ========== Common Event Preparation ==========

"""
    prepare_event_for_emd(event::Matrix{Float64}; params::EMDParameters)

Prepare event for EMD computation with standard preprocessing.
"""
function prepare_event_for_emd(event::Matrix{Float64}; params::EMDParameters)
    processed = copy(event)

    # Apply masking if requested
    if params.mask
        processed = mask_particles_outside_radius(processed, params.R)
    end

    # Normalize weights if requested
    if params.norm && size(processed, 1) > 0
        total_weight = sum(processed[:, 1])
        if total_weight > 0
            processed[:, 1] ./= total_weight
        end
    end

    return processed
end

export prepare_event_for_emd

# ========== Format Conversion ==========

"""
    format_event(weights::AbstractVector, coordinates::AbstractMatrix)

Format weights and coordinates into an event matrix for EMD computation.
"""
function format_event(weights::AbstractVector, coordinates::AbstractMatrix)
    if length(weights) != size(coordinates, 2)
        throw(DimensionMismatch("Number of weights must match number of particles"))
    end

    # Create matrix with weights in first column
    result = zeros(eltype(weights), length(weights), 1 + size(coordinates, 1))
    result[:, 1] = weights
    result[:, 2:end] = coordinates'

    return result
end

export format_event

end # module Utils
