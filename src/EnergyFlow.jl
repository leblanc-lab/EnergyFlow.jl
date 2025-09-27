# EnergyFlow.jl - Main module for the EnergyFlow package

module EnergyFlow

using LinearAlgebra
using JuMP
using HiGHS
using LoopVectorization
using StaticArrays
using Base.Threads

# Include shared utilities first
include("Utils.jl")
using .Utils

# Include distance computation
include("Distances.jl")
using .Distances

# Include EMD module with backend system
include("EMD.jl")
using .EMD

# Re-export main functions from EMD
export emd, emds
export set_backend, get_backend
export emd_network_simplex, emds_network_simplex
export emd_exact, emds_exact
export emd_cxx, emds_cxx

# Re-export utilities
export EMDParameters
export format_event, prepare_event_for_emd
export cartesian_to_hadronic, apply_kinematic_cuts
export mask_particles_outside_radius
export total_momentum, center_of_energy

# Re-export distance functions
export compute_distance_matrix, pairwise_distances
export euclidean_distance_2d, euclidean_distance_3d, periodic_phi_distance

end # module EnergyFlow