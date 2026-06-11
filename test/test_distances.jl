using Test
using EnergyFlow
println("="^70)
println("DISTANCES TESTS")
println("="^70)
@testset "Distances" begin
    # Euclidean (1D): distance = 3 -> cost = 3 (beta=1, R=1)
    ws = EMDWorkspace(1, 1; beta=1.0, R=1.0)
    coords0 = reshape([0.0], 1, 1)
    coords1 = reshape([3.0], 1, 1)
    EnergyFlow._fill_costs!(ws, EuclideanMetric(), coords0, coords1, 1, 1, false, false)
        println("  Euclidean cost = $(ws.ns.costs[1])")
    @test ws.ns.costs[1] == 3.0

    # SquaredEuclidean: with beta=1 and R=1 cost = d^2 = 9
    ws2 = EMDWorkspace(1, 1; beta=1.0, R=1.0)
    EnergyFlow._fill_costs!(ws2, SquaredEuclideanMetric(), coords0, coords1, 1, 1, false, false)
        println("  SquaredEuclidean cost = $(ws2.ns.costs[1])")
    @test ws2.ns.costs[1] == 9.0

    # CustomMetric: Manhattan distance, beta=2 -> (3)^2 = 9
    manhattan(ci, cj) = sum(abs.(ci .- cj))
    ws3 = EMDWorkspace(1, 1; beta=2.0, R=1.0)
    EnergyFlow._fill_costs!(ws3, CustomMetric(manhattan), coords0, coords1, 1, 1, false, false)
        println("  CustomMetric cost = $(ws3.ns.costs[1])")
    @test ws3.ns.costs[1] == 9.0

    # PrecomputedMetric: values are used as-is
    pm = reshape([5.0], 1, 1)
    ws4 = EMDWorkspace(1, 1)
    EnergyFlow._fill_costs!(ws4, PrecomputedMetric(pm), coords0, coords1, 1, 1, false, false)
        println("  PrecomputedMetric cost = $(ws4.ns.costs[1])")
    @test ws4.ns.costs[1] == 5.0

    # EtaPhi periodicity: phi wrap near +/-pi should produce small delta
    coords0 = zeros(Float64, 2, 1)
    coords1 = zeros(Float64, 2, 1)
    coords0[1, 1] = 0.0; coords1[1, 1] = 0.0
    coords0[2, 1] = π - 0.1; coords1[2, 1] = -π + 0.1
    ws5 = EMDWorkspace(1, 1; beta=1.0, R=1.0)
    EnergyFlow._fill_costs!(ws5, EtaPhiMetric(), coords0, coords1, 1, 1, false, false)
        println("  EtaPhi cost = $(ws5.ns.costs[1])")
    @test isapprox(ws5.ns.costs[1], 0.2; atol=1e-12)

    # Fictitious particle costs should be set to one(V)
    coords0 = reshape([0.0], 1, 1)
    coords1 = reshape([0.0], 1, 1)
    ws6 = EMDWorkspace(2, 2; beta=1.0, R=1.0)
    EnergyFlow._fill_costs!(ws6, EuclideanMetric(), coords0, coords1, 2, 2, true, true)
        println("  Fictitious costs = $(ws6.ns.costs[3]), $(ws6.ns.costs[4])")
    # index 3 = (n0_eff-1)*n1_eff + 1 -> should be one
    @test ws6.ns.costs[3] == one(Float64)
end
println("\nAll Distances tests completed!")