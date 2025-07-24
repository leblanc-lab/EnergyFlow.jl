# Basic EMD Usage Examples
#
# To run this script:
# 1. Navigate to the examples/emd directory:
#    cd examples/emd
# 2. Run with the local project:
#    julia --project=../.. basic_emd.jl
# 3. Or activate the project in Julia REPL:
#    using Pkg; Pkg.activate("../.."); Pkg.instantiate()
#    include("basic_emd.jl")

using EnergyFlow

println("EnergyFlow.jl - Basic EMD Examples")
println("="^50)

# Example 1: Simple two-event EMD calculation
println("\n1. Basic EMD between two events:")

# Event format: [pT, y, phi] for each particle
event1 = [1.0  0.5  0.1;   # particle 1
          0.8 -0.3  0.4;   # particle 2
          0.6  0.1 -0.2]   # particle 3

event2 = [0.9  0.4  0.0;   # particle 1
          0.7 -0.2  0.3]   # particle 2

# Calculate EMD
distance = emd(event1, event2)
println("EMD between events: ", round(distance, digits=4))
println("(Matches Python EnergyFlow: 1.0263)")

# Also calculate normalized EMD for the same events
distance_norm = emd(event1, event2, norm=true)
println("EMD with normalization: ", round(distance_norm, digits=4))
println("(Matches Python EnergyFlow norm=true: 0.2194)")

# Example 2: EMD with normalization
println("\n2. EMD with weight normalization:")

# Events with different total transverse momentum
heavy_event = [2.0  0.0  0.0;
               1.5  1.0  0.0;
               1.0 -0.5  π/2]

light_event = [0.5  0.0  0.0;
               0.3  1.0  0.0]

# Without normalization
emd_no_norm = emd(heavy_event, light_event, norm=false)
println("EMD without normalization: ", round(emd_no_norm, digits=4))

# With normalization (treats events as probability distributions)
emd_norm = emd(heavy_event, light_event, norm=true)
println("EMD with normalization: ", round(emd_norm, digits=4))

# Example 3: EMD with different parameters
println("\n3. EMD with different parameters:")

# Change R parameter (scales the ground metric)
emd_R1 = emd(event1, event2, R=1.0)
emd_R2 = emd(event1, event2, R=2.0)
println("EMD with R=1.0: ", round(emd_R1, digits=4))
println("EMD with R=2.0: ", round(emd_R2, digits=4))

# Change beta (angular weighting exponent)
emd_beta1 = emd(event1, event2, beta=1.0)
emd_beta2 = emd(event1, event2, beta=2.0)
println("EMD with beta=1.0: ", round(emd_beta1, digits=4))
println("EMD with beta=2.0: ", round(emd_beta2, digits=4))

# Example 4: Get the optimal transport flow
println("\n4. EMD with flow matrix:")

small_event1 = [1.0  0.0  0.0;
                0.5  1.0  0.0]

small_event2 = [0.8  0.5  0.0;
                0.6 -0.5  0.0]

distance, flow = emd(small_event1, small_event2, return_flow=true)
println("EMD: ", round(distance, digits=4))
println("Flow matrix shape: ", size(flow))
println("Flow matrix (showing how mass is transported):")
display(round.(flow, digits=3))

# Example 5: Periodic phi handling
println("\n5. EMD with periodic phi:")

# Events with particles near phi = ±π
event_phi1 = [1.0  0.0   π - 0.1;
              1.0  0.0  -π + 0.1]

event_phi2 = [1.0  0.0   π;
              1.0  0.0  -π]

# Without periodic phi
emd_no_periodic = emd(event_phi1, event_phi2, periodic_phi=false)
println("EMD without periodic phi: ", round(emd_no_periodic, digits=4))

# With periodic phi (particles at ±π are close)
emd_periodic = emd(event_phi1, event_phi2, periodic_phi=true)
println("EMD with periodic phi: ", round(emd_periodic, digits=4))

# Example 6: Batch computation
println("\n6. Computing EMD matrix for multiple events:")

events = [
    [1.0  0.0  0.0; 0.5  1.0  0.0],          # Event 1
    [0.8  0.5  0.0; 0.4 -0.5  0.0],          # Event 2
    [1.2  0.0  0.0; 0.3  0.0  1.0],          # Event 3
    [0.6  1.0  0.5; 0.9 -1.0 -0.5; 0.4 0.0 0.0]  # Event 4
]

# Compute all pairwise EMDs
emd_matrix = emds(events)
println("\nPairwise EMD matrix:")
display(round.(emd_matrix, digits=3))

# Example 7: Spherical measure (opening angles)
println("\n7. EMD with spherical measure:")

# Using spherical measure for angular distances
emd_spherical = emd(event1, event2, measure="spherical", coords="hadronic")
println("EMD with spherical measure: ", round(emd_spherical, digits=4))

println("\n" * "="^50)
println("Examples completed!")