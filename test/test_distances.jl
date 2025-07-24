@testset "Distance Function Tests" begin
    
    @testset "Euclidean Distance Matrix" begin
        # Simple 2D coordinates
        coords1 = [0.0 0.0;
                   1.0 0.0;
                   0.0 1.0]
        
        coords2 = [0.0 0.0;
                   1.0 1.0]
        
        params = EnergyFlow.EMDParameters(measure="euclidean", R=1.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(coords1, coords2, params)
        
        @test size(dist_matrix) == (3, 2)
        @test dist_matrix[1, 1] ≈ 0.0      # (0,0) to (0,0)
        @test dist_matrix[2, 1] ≈ 1.0      # (1,0) to (0,0)
        @test dist_matrix[3, 2] ≈ 1.0      # (0,1) to (1,1)
        @test dist_matrix[1, 2] ≈ √2       # (0,0) to (1,1)
    end
    
    @testset "Distance Matrix with R scaling" begin
        coords1 = [0.0 0.0;
                   2.0 0.0]
        
        coords2 = [0.0 0.0;
                   0.0 2.0]
        
        # Test with R=2.0
        params = EnergyFlow.EMDParameters(measure="euclidean", R=2.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(coords1, coords2, params)
        
        @test dist_matrix[2, 1] ≈ 2.0 / 2.0  # Distance 2, scaled by R
        @test dist_matrix[2, 2] ≈ 2√2 / 2.0  # Distance 2√2, scaled by R
    end
    
    @testset "Periodic Phi Distance" begin
        # Test periodic boundary in phi
        coords1 = [0.0 0.0;      # y=0, phi=0
                   0.0 π-0.1;    # y=0, phi≈π
                   0.0 -π+0.1]   # y=0, phi≈-π
        
        coords2 = [0.0 0.0;      # y=0, phi=0
                   0.0 π;        # y=0, phi=π
                   0.0 -π]       # y=0, phi=-π
        
        params = EnergyFlow.EMDParameters(measure="euclidean", periodic_phi=true, R=1.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(coords1, coords2, params)
        
        # Points at ±π should be close with periodic phi
        @test dist_matrix[2, 3] < 0.3  # (π-0.1) to -π
        @test dist_matrix[3, 2] < 0.3  # (-π+0.1) to π
        
        # Without periodic phi, they should be far
        params_no_periodic = EnergyFlow.EMDParameters(measure="euclidean", periodic_phi=false, R=1.0)
        dist_matrix_no_periodic = EnergyFlow.compute_distance_matrix(coords1, coords2, params_no_periodic)
        
        @test dist_matrix_no_periodic[2, 3] > 2.0  # Should be ≈ 2π apart
    end
    
    @testset "Spherical Distance (Opening Angles)" begin
        # Test spherical measure with 3D unit vectors
        coords1 = [1.0 0.0 0.0;    # x-axis
                   0.0 1.0 0.0;    # y-axis
                   0.0 0.0 1.0]    # z-axis
        
        coords2 = [1.0 0.0 0.0;                      # x-axis
                   1/√2 1/√2 0.0;                    # 45° between x and y
                   1/√3 1/√3 1/√3]                   # Equal angles
        
        params = EnergyFlow.EMDParameters(measure="spherical", R=1.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(coords1, coords2, params)
        
        @test dist_matrix[1, 1] ≈ 0.0           # Same direction
        @test dist_matrix[1, 2] ≈ π/4           # 45° angle
        @test dist_matrix[2, 2] ≈ π/4           # y-axis to 45° vector
        @test dist_matrix[1, 3] ≈ acos(1/√3)    # x-axis to equal-angle vector
        
        # Orthogonal vectors should be π/2 apart
        @test abs(dist_matrix[1, 1] - 0) < 1e-10
        @test abs(dist_matrix[2, 1] - π/2) < 1e-10
    end
    
    @testset "Pairwise Distances" begin
        # Test symmetric distance computation
        coords = [0.0 0.0;
                  1.0 0.0;
                  0.0 1.0;
                  1.0 1.0]
        
        params = EnergyFlow.EMDParameters(measure="euclidean", R=1.0)
        pairwise_dist = EnergyFlow.pairwise_distances(coords, params)
        
        @test size(pairwise_dist) == (4, 4)
        @test issymmetric(pairwise_dist)
        @test all(diag(pairwise_dist) .≈ 0.0)
        
        # Check specific distances
        @test pairwise_dist[1, 2] ≈ 1.0    # (0,0) to (1,0)
        @test pairwise_dist[1, 4] ≈ √2     # (0,0) to (1,1)
        @test pairwise_dist[2, 3] ≈ √2     # (1,0) to (0,1)
    end
    
    @testset "Edge Cases for Distances" begin
        # Single point
        coords1 = reshape([0.0, 0.0], 1, 2)
        coords2 = reshape([1.0, 1.0], 1, 2)
        
        params = EnergyFlow.EMDParameters(measure="euclidean", R=1.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(coords1, coords2, params)
        
        @test size(dist_matrix) == (1, 1)
        @test dist_matrix[1, 1] ≈ √2
        
        # Empty coordinates
        empty_coords = zeros(0, 2)
        dist_empty = EnergyFlow.compute_distance_matrix(empty_coords, coords1, params)
        @test size(dist_empty) == (0, 1)
        
        # High dimensional
        coords_5d = rand(10, 5)
        dist_5d = EnergyFlow.compute_distance_matrix(coords_5d, coords_5d, params)
        @test size(dist_5d) == (10, 10)
        @test all(diag(dist_5d) .< 1e-10)
    end
    
    @testset "Distance Numerical Stability" begin
        # Very close points
        coords1 = [0.0 0.0]
        coords2 = [1e-15 1e-15]
        
        params = EnergyFlow.EMDParameters(measure="euclidean", R=1.0)
        dist_matrix = EnergyFlow.compute_distance_matrix(
            reshape(coords1, 1, 2), 
            reshape(coords2, 1, 2), 
            params
        )
        
        @test dist_matrix[1, 1] < 1e-10
        @test dist_matrix[1, 1] >= 0.0
        
        # Large coordinates
        coords_large = [1e6 1e6;
                       1e6 + 1 1e6]
        
        dist_large = EnergyFlow.compute_distance_matrix(coords_large, coords_large, params)
        @test isfinite(dist_large[1, 2])
        @test dist_large[1, 2] ≈ 1.0  # Should still get distance of 1
    end
end