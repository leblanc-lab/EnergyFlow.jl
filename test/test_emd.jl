@testset "EMD Tests" begin
    @testset "Basic EMD Computation" begin
        event1 = [1.0 0.5 0.1;
                  0.8 -0.3 0.4;
                  0.6 0.1 -0.2]
        
        event2 = [0.9 0.4 0.0;
                  0.7 -0.2 0.3]
        
        emd_val = emd(event1, event2)
        @test emd_val isa Float64
        @test emd_val >= 0
        
        emd_val2, flow = emd(event1, event2, return_flow=true)
        @test emd_val ≈ emd_val2
        @test size(flow) == (4, 3)
        @test all(flow .>= -1e-10)

        self_emd = emd(event1, event1)
        @test self_emd < 1e-10
    end
    
    @testset "EMD with Normalization" begin
        event1 = [2.0 0.0 0.0;
                  1.0 1.0 0.0]
        
        event2 = [1.0 0.0 0.0;
                  0.5 1.0 0.0]
        
        emd_no_norm = emd(event1, event2, norm=false)
        
        emd_norm = emd(event1, event2, norm=true)
        
        @test emd_no_norm > emd_norm
        @test emd_norm >= 0
    end
    
    @testset "EMD with Different Parameters" begin
        event1 = [1.0 0.5 0.1;
                  0.8 -0.3 0.4]
        
        event2 = [0.9 0.4 0.0;
                  0.7 -0.2 0.3]
        
        emd_r1 = emd(event1, event2, R=1.0)
        emd_r2 = emd(event1, event2, R=2.0)
        @test emd_r1 != emd_r2
        
        emd_beta1 = emd(event1, event2, beta=1.0)
        emd_beta2 = emd(event1, event2, beta=2.0)
        @test emd_beta1 != emd_beta2
        
        emd_periodic = emd(event1, event2, periodic_phi=true)
        @test emd_periodic isa Float64
    end
    
    @testset "EMD with Empty Events" begin
        empty_event = zeros(0, 3)
        event = [1.0 0.0 0.0]
        
        @test emd(empty_event, empty_event) < 1e-10
        
        @test_throws Exception emd(empty_event, event)
    end
    
    @testset "EMD Matrix Computation" begin
        events = [
            [1.0 0.0 0.0; 0.5 1.0 0.0],
            [0.8 0.5 0.0; 0.4 -0.5 0.0],
            [1.2 0.0 0.0; 0.3 0.0 1.0]
        ]
        
        emd_matrix = emds(events)
        
        @test size(emd_matrix) == (3, 3)
        @test issymmetric(emd_matrix)
        @test all(diag(emd_matrix) .< 1e-10)
        @test all(emd_matrix .>= -1e-10)
    end
    
    @testset "EMD with Cartesian Coordinates" begin
        event1_cart = [2.0 1.0 0.0 1.5;
                       1.5 0.0 1.0 0.8]
        
        event2_cart = [1.8 0.8 0.5 1.2;
                       1.2 0.5 0.8 0.3]
        
        emd_cart = emd(event1_cart, event2_cart, coords="cartesian")
        @test emd_cart isa Float64
        @test emd_cart >= 0
    end
    
    @testset "Event Preparation" begin
        raw_event = [1.0 0.5 0.1 1.2;
                     0.8 -0.3 0.4 0.9;
                     0.6 0.1 -0.2 1.1]
        
        prepared = prepare_event_for_emd(raw_event)
        @test size(prepared) == (3, 3)
        @test prepared[:, 1] == raw_event[:, 1]  # pT
        @test prepared[:, 2] == raw_event[:, 2]  # y
        @test prepared[:, 3] == raw_event[:, 3]  # phi
        
        prepared_w = prepare_event_for_emd(raw_event, weight_col=4)
        @test size(prepared_w) == (3, 4)
        @test prepared_w[:, 4] == raw_event[:, 4]
    end
    
    @testset "Numerical Stability" begin
        event1 = [1e-10 0.0 0.0;
                  1e-10 1.0 0.0]
        
        event2 = [1e-10 0.0 0.0;
                  1e-10 1.0 0.0]
        
        emd_small = emd(event1, event2)
        @test isfinite(emd_small)
        @test emd_small >= 0
        
        event1_large = 1e6 * event1
        event2_large = 1e6 * event2
        
        emd_large = emd(event1_large, event2_large)
        @test isfinite(emd_large)
        @test emd_large >= 0
    end
    
    @testset "Edge Cases" begin
        single1 = reshape([1.0, 0.0, 0.0], 1, 3)
        single2 = reshape([1.0, 1.0, 0.0], 1, 3)
        
        emd_single = emd(single1, single2)
        @test emd_single ≈ 1.0 
        
        event1 = [1.0 0.0 0.0;
                  2.0 0.0 0.0]
        
        event2 = [3.0 0.0 0.0]
        
        emd_same_loc = emd(event1, event2)
        @test emd_same_loc < 1e-10
    end
end