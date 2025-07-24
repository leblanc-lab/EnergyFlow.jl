@testset "Utility Function Tests" begin
    
    @testset "Coordinate Transformations" begin
        # Test cartesian to hadronic conversion
        cart_event = [5.0 3.0 4.0 0.0;    # E=5, px=3, py=4, pz=0
                      2.0 0.0 0.0 1.732]   # E=2, px=0, py=0, pz≈√3
        
        hadr_event = EnergyFlow.cartesian_to_hadronic(cart_event)
        
        # Check pT calculation
        @test hadr_event[1, 1] ≈ 5.0  # sqrt(3^2 + 4^2) = 5
        @test hadr_event[2, 1] ≈ 0.0  # sqrt(0^2 + 0^2) = 0
        
        # Check phi calculation
        @test hadr_event[1, 3] ≈ atan(4.0, 3.0)
        
        # Test hadronic to cartesian conversion
        hadr_input = [1.0 0.5 0.0;    # pT=1, y=0.5, phi=0
                      2.0 0.0 π/2]    # pT=2, y=0, phi=π/2
        
        cart_output = EnergyFlow.hadronic_to_cartesian(hadr_input)
        
        # Check conversions
        @test cart_output[1, 2] ≈ 1.0 * cos(0)      # px
        @test cart_output[1, 3] ≈ 1.0 * sin(0)      # py
        @test cart_output[2, 2] ≈ 2.0 * cos(π/2)    # px ≈ 0
        @test cart_output[2, 3] ≈ 2.0 * sin(π/2)    # py ≈ 2
    end
    
    @testset "Kinematic Cuts" begin
        # Test event with various pT and y values
        event = [0.5 0.1 0.0;    # Low pT
                 2.0 5.5 0.0;    # High y
                 1.5 0.5 0.0;    # Pass cuts
                 0.1 0.0 0.0;    # Low pT
                 3.0 -6.0 0.0]   # High |y|
        
        # Apply cuts
        cut_event = EnergyFlow.apply_kinematic_cuts(event, pT_min=1.0, y_max=5.0)
        
        @test size(cut_event, 1) == 1  # Only one particle passes
        @test cut_event[1, 1] ≈ 1.5   # The third particle
        
        # Test with cartesian coordinates
        cart_event = [5.0 3.0 4.0 0.0;
                      1.0 0.1 0.1 0.8]
        
        cut_cart = EnergyFlow.apply_kinematic_cuts(cart_event, pT_min=1.0, 
                                                   coords="cartesian")
        @test size(cut_cart, 1) == 1  # Only first particle has pT > 1
    end
    
    @testset "Particle Masking" begin
        # Test masking by radius
        event = [1.0 0.0 0.0;      # At origin
                 1.0 1.0 0.0;      # Distance = 1
                 1.0 0.0 2.0;      # Distance = 2
                 1.0 1.5 1.5]      # Distance ≈ 2.12
        
        masked = EnergyFlow.mask_particles_outside_radius(event, 1.5)
        @test size(masked, 1) == 2  # First two particles
        
        masked2 = EnergyFlow.mask_particles_outside_radius(event, 2.5)
        @test size(masked2, 1) == 4  # All particles
    end
    
    @testset "Momentum Calculations" begin
        # Test total momentum calculation
        hadr_event = [1.0 0.0 0.0;     # pT=1, y=0, phi=0
                      2.0 0.0 π/2]     # pT=2, y=0, phi=π/2
        
        total_p = EnergyFlow.total_momentum(hadr_event)
        
        @test length(total_p) == 4
        @test total_p[1] ≈ 3.0        # Total E for massless particles
        @test total_p[2] ≈ 1.0        # Total px
        @test total_p[3] ≈ 2.0        # Total py
        @test abs(total_p[4]) < 1e-10 # Total pz ≈ 0
        
        # Test with cartesian
        cart_event = [3.0 1.0 2.0 0.0;
                      2.0 0.0 1.0 1.0]
        
        total_p_cart = EnergyFlow.total_momentum(cart_event, coords="cartesian")
        @test total_p_cart ≈ [5.0, 1.0, 3.0, 1.0]
    end
    
    @testset "Center of Energy" begin
        # Test center of energy calculation
        event = [2.0 1.0 0.0;    # Heavy particle at y=1
                 1.0 -1.0 0.0]   # Light particle at y=-1
        
        center = EnergyFlow.center_of_energy(event)
        
        @test center[1] ≈ 3.0                    # Total pT
        @test center[2] ≈ (2*1 + 1*(-1))/3      # Weighted y
        @test abs(center[3]) < 1e-10            # phi ≈ 0
        
        # Test with different phi values
        event_phi = [1.0 0.0 0.0;
                     1.0 0.0 π;
                     1.0 0.0 π/2]
        
        center_phi = EnergyFlow.center_of_energy(event_phi)
        # Should handle circular average correctly
        @test center_phi[1] ≈ 3.0
    end
    
    @testset "Edge Cases for Utils" begin
        # Empty event
        empty_event = zeros(0, 3)
        
        # Should handle empty events gracefully
        cut_empty = EnergyFlow.apply_kinematic_cuts(empty_event)
        @test size(cut_empty, 1) == 0
        
        masked_empty = EnergyFlow.mask_particles_outside_radius(empty_event, 1.0)
        @test size(masked_empty, 1) == 0
        
        # Zero energy event
        zero_event = [0.0 0.0 0.0;
                      0.0 1.0 0.0]
        
        center_zero = EnergyFlow.center_of_energy(zero_event)
        @test all(center_zero .== 0.0)
    end
    
    @testset "Coordinate Normalization" begin
        # Test coordinate normalization for spherical measure
        coords = [3.0 4.0;    # Length = 5
                  0.0 0.0;    # Zero vector
                  1.0 0.0]    # Length = 1
        
        norm_coords = EnergyFlow.normalize_coordinates(coords)
        
        @test norm_coords[1, :] ≈ [3/5, 4/5]
        @test norm_coords[2, :] ≈ [0.0, 0.0]  # Unchanged
        @test norm_coords[3, :] ≈ [1.0, 0.0]
        
        # Check all non-zero vectors have unit length
        for i in [1, 3]
            @test norm(norm_coords[i, :]) ≈ 1.0
        end
    end
end