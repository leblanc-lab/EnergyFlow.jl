# Events.jl - Event types and manipulation functions shared across solvers

module Events

export AbstractEvent, Event, EuclideanParticleEvent
export total_weight, normalize_weights!, ensure_weights, dimension, n_particles

# Abstract base type for events
abstract type AbstractEvent{V<:AbstractFloat} end

# Basic Event structure
mutable struct Event{V<:AbstractFloat} <: AbstractEvent{V}
    particles::Matrix{V}  # Each column is a particle
    weights::Vector{V}
    event_weight::V
    total_weight::V
    has_weights::Bool

    # Constructor from particles only
    function Event(particles::Matrix{V}, event_weight::V=one(V)) where V
        n_particles = size(particles, 2)
        new{V}(particles, Vector{V}(), event_weight, zero(V), false)
    end

    # Constructor from weights and particles
    function Event(weights::Vector{V}, particles::Matrix{V}, event_weight::V=one(V)) where V
        if length(weights) != size(particles, 2)
            throw(DimensionMismatch("Number of weights must match number of particles"))
        end
        total = sum(weights)
        new{V}(particles, weights, event_weight, total, true)
    end
end

# Euclidean Particle Event (particles with weights embedded)
mutable struct EuclideanParticleEvent{V<:AbstractFloat} <: AbstractEvent{V}
    particles::Matrix{V}
    weights::Vector{V}
    event_weight::V
    total_weight::V
    has_weights::Bool

    function EuclideanParticleEvent(particles::Matrix{V}, event_weight::V=one(V)) where V
        n_particles = size(particles, 2)
        # Extract weights from first row of particles (common in physics applications)
        weights = particles[1, :]
        total = sum(weights)
        new{V}(particles, weights, event_weight, total, true)
    end
end

# Helper functions
function total_weight(ev::AbstractEvent)
    if !ev.has_weights
        throw(ErrorException("Event does not have weights"))
    end
    return ev.total_weight
end

function ensure_weights(ev::AbstractEvent{V}) where V
    if !ev.has_weights
        # Default: equal weights
        n_particles = size(ev.particles, 2)
        ev.weights = ones(V, n_particles) / n_particles
        ev.total_weight = one(V)
        ev.has_weights = true
    end
end

function normalize_weights!(ev::AbstractEvent{V}) where V
    if !ev.has_weights
        throw(ErrorException("Weights must be set prior to normalization"))
    end

    if ev.total_weight ≈ 0
        throw(ErrorException("Cannot normalize zero total weight"))
    end

    # Normalize each weight
    norm_total = zero(V)
    for i in eachindex(ev.weights)
        ev.weights[i] /= ev.total_weight
        norm_total += ev.weights[i]
    end
    ev.total_weight = norm_total
end

# Dimension of particles
dimension(ev::AbstractEvent) = size(ev.particles, 1)

# Number of particles
n_particles(ev::AbstractEvent) = size(ev.particles, 2)

end # module Events