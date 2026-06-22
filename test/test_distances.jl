using Test
using EnergyFlow
include("test_helpers.jl")
test_log("="^70)
test_log("DISTANCES TESTS")
test_log("="^70)

distance_test_workspace(beta, R; norm=true) = EMDWorkspace(1, 1; beta=beta, R=R, norm=norm)

@testset "Distances" begin
    # Euclidean (1D): distance = 3 -> cost = 3 (beta=1, R=1)
    euclidean_coords0 = reshape([0.0], 1, 1)
    euclidean_coords1 = reshape([3.0], 1, 1)
    euclidean_ws = distance_test_workspace(1.0, 1.0)
    EnergyFlow._fill_costs!(euclidean_ws, EuclideanMetric(), euclidean_coords0, euclidean_coords1, 1, 1, false, false)
    test_log("  Euclidean cost = $(euclidean_ws.ns.costs[1])")
    @test euclidean_ws.ns.costs[1] == 3.0

    # SquaredEuclidean: with beta=1 and R=1 cost = d^2 = 9
    squared_coords0 = reshape([0.0], 1, 1)
    squared_coords1 = reshape([3.0], 1, 1)
    squared_ws = distance_test_workspace(1.0, 1.0)
    EnergyFlow._fill_costs!(squared_ws, SquaredEuclideanMetric(), squared_coords0, squared_coords1, 1, 1, false, false)
    test_log("  SquaredEuclidean cost = $(squared_ws.ns.costs[1])")
    @test squared_ws.ns.costs[1] == 9.0

    # CustomMetric: Manhattan distance, beta=2 -> (3)^2 = 9
    manhattan(ci, cj) = sum(abs.(ci .- cj))
    custom_coords0 = reshape([0.0], 1, 1)
    custom_coords1 = reshape([3.0], 1, 1)
    custom_ws = distance_test_workspace(2.0, 1.0)
    EnergyFlow._fill_costs!(custom_ws, CustomMetric(manhattan), custom_coords0, custom_coords1, 1, 1, false, false)
    test_log("  CustomMetric cost = $(custom_ws.ns.costs[1])")
    @test custom_ws.ns.costs[1] == 9.0

    # PrecomputedMetric: values are used as-is
    pm = reshape([5.0], 1, 1)
    precomputed_coords0 = reshape([0.0], 1, 1)
    precomputed_coords1 = reshape([3.0], 1, 1)
    precomputed_ws = EMDWorkspace(1, 1)
    EnergyFlow._fill_costs!(precomputed_ws, PrecomputedMetric(pm), precomputed_coords0, precomputed_coords1, 1, 1, false, false)
    test_log("  PrecomputedMetric cost = $(precomputed_ws.ns.costs[1])")
    @test precomputed_ws.ns.costs[1] == 5.0

    # EtaPhi periodicity: phi wrap near +/-pi should produce small delta
    etaphi_coords0 = zeros(Float64, 2, 1)
    etaphi_coords1 = zeros(Float64, 2, 1)
    etaphi_coords0[1, 1] = 0.0; etaphi_coords1[1, 1] = 0.0
    etaphi_coords0[2, 1] = π - 0.1; etaphi_coords1[2, 1] = -π + 0.1
    etaphi_ws = distance_test_workspace(1.0, 1.0)
    EnergyFlow._fill_costs!(etaphi_ws, EtaPhiMetric(), etaphi_coords0, etaphi_coords1, 1, 1, false, false)
    test_log("  EtaPhi cost = $(etaphi_ws.ns.costs[1])")
    @test isapprox(etaphi_ws.ns.costs[1], 0.2; atol=1e-12)

    # Fictitious particle costs should be set to one(V)
    fictitious_coords0 = reshape([0.0], 1, 1)
    fictitious_coords1 = reshape([0.0], 1, 1)
    fictitious_ws = EMDWorkspace(2, 2; beta=1.0, R=1.0)
    EnergyFlow._fill_costs!(fictitious_ws, EuclideanMetric(), fictitious_coords0, fictitious_coords1, 2, 2, true, true)
    test_log("  Fictitious costs = $(fictitious_ws.ns.costs[3]), $(fictitious_ws.ns.costs[4])")
    # index 3 = (n0_eff-1)*n1_eff + 1 -> should be one
    @test fictitious_ws.ns.costs[3] == one(Float64)
end
test_log("\nAll Distances tests completed!")
