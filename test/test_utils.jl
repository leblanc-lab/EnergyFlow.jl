@testset "Utility Function Tests" begin
    
    @testset "Coordinate Transformations" begin
        cart_event = [5.0 3.0 4.0 0.0;
                      2.0 0.0 0.0 1.732]
        
        hadr_event = EnergyFlow.cartesian_to_hadronic(cart_event)
        
        @test hadr_event[1, 1] ≈ 5.0
        @test hadr_event[2, 1] ≈ 0.0
        @test hadr_event[1, 3] ≈ atan(4.0, 3.0)
        
        hadr_input = [1.0 0.5 0.0;
                      2.0 0.0 π/2]
        
        cart_output = EnergyFlow.hadronic_to_cartesian(hadr_input)
        
        # Check conversions
        @test cart_output[1, 2] ≈ 1.0 * cos(0)
        @test cart_output[1, 3] ≈ 1.0 * sin(0)
        @test cart_output[2, 2] ≈ 2.0 * cos(π/2)
        @test cart_output[2, 3] ≈ 2.0 * sin(π/2)
    end
    
    @testset "Kinematic Cuts" begin
        event = [0.5 0.1 0.0;    # Low pT
                 2.0 5.5 0.0;    # High y
                 1.5 0.5 0.0;    # Pass cuts
                 0.1 0.0 0.0;    # Low pT
                 3.0 -6.0 0.0]   # High |y|
        
        cut_event = EnergyFlow.apply_kinematic_cuts(event, pT_min=1.0, y_max=5.0)
        
        @test size(cut_event, 1) == 1
        @test cut_event[1, 1] ≈ 1.5
        
        cart_event = [5.0 3.0 4.0 0.0;
                      1.0 0.1 0.1 0.8]
        
        cut_cart = EnergyFlow.apply_kinematic_cuts(cart_event, pT_min=1.0, 
                                                   coords="cartesian")
        @test size(cut_cart, 1) == 1
    end
    
    @testset "Particle Masking" begin
        event = [1.0 0.0 0.0;
                 1.0 1.0 0.0;
                 1.0 0.0 2.0;
                 1.0 1.5 1.5]
        
        masked = EnergyFlow.mask_particles_outside_radius(event, 1.5)
        @test size(masked, 1) == 2
        
        masked2 = EnergyFlow.mask_particles_outside_radius(event, 2.5)
        @test size(masked2, 1) == 4
    end
    
    @testset "Momentum Calculations" begin
        hadr_event = [1.0 0.0 0.0;
                      2.0 0.0 π/2]
        
        total_p = EnergyFlow.total_momentum(hadr_event)
        
        @test length(total_p) == 4
        @test total_p[1] ≈ 3.0
        @test total_p[2] ≈ 1.0
        @test total_p[3] ≈ 2.0
        @test abs(total_p[4]) < 1e-10
        
        cart_event = [3.0 1.0 2.0 0.0;
                      2.0 0.0 1.0 1.0]
        
        total_p_cart = EnergyFlow.total_momentum(cart_event, coords="cartesian")
        @test total_p_cart ≈ [5.0, 1.0, 3.0, 1.0]
    end
    
    @testset "Center of Energy" begin
        event = [2.0 1.0 0.0;
                 1.0 -1.0 0.0]
        
        center = EnergyFlow.center_of_energy(event)
        
        @test center[1] ≈ 3.0
        @test center[2] ≈ (2*1 + 1*(-1))/3
        @test abs(center[3]) < 1e-10
        
        event_phi = [1.0 0.0 0.0;
                     1.0 0.0 π;
                     1.0 0.0 π/2]
        
        center_phi = EnergyFlow.center_of_energy(event_phi)
        @test center_phi[1] ≈ 3.0
    end
    
    @testset "Edge Cases for Utils" begin
        empty_event = zeros(0, 3)
        
        cut_empty = EnergyFlow.apply_kinematic_cuts(empty_event)
        @test size(cut_empty, 1) == 0
        
        masked_empty = EnergyFlow.mask_particles_outside_radius(empty_event, 1.0)
        @test size(masked_empty, 1) == 0
        
        zero_event = [0.0 0.0 0.0;
                      0.0 1.0 0.0]
        
        center_zero = EnergyFlow.center_of_energy(zero_event)
        @test all(center_zero .== 0.0)
    end
    
    @testset "Coordinate Normalization" begin
        coords = [3.0 4.0;
                  0.0 0.0;
                  1.0 0.0]
        
        norm_coords = EnergyFlow.normalize_coordinates(coords)
        
        @test norm_coords[1, :] ≈ [3/5, 4/5]
        @test norm_coords[2, :] ≈ [0.0, 0.0]
        @test norm_coords[3, :] ≈ [1.0, 0.0]
        
        for i in [1, 3]
            @test norm(norm_coords[i, :]) ≈ 1.0
        end
    end
end