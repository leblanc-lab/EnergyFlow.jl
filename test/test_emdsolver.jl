using Test
using EnergyFlow
include("test_helpers.jl")

test_log("="^70)
test_log("EMDSOLVER TESTS")
test_log("="^70)

@testset "EMDSolver core" begin
    test_log("  core: constructing NetworkSimplexSolver{Float64}(1, 1)")
    ns = NetworkSimplexSolver{Float64}(1, 1)
    ns.costs[1] = 3.0

    status = network_simplex!(ns, [1.0], [1.0])
    test_log("  core: status=$status total_cost=$(ns.total_cost)")
    @test status == :optimal
    @test ns.total_cost == 3.0
    @test ns.n0 == 1
    @test ns.n1 == 1
    @test ns.arc_num == 1
end

@testset "EMDSolver raw solve" begin
    test_log("  raw solve: balanced 1x1 transport with distance 3")
    ws = EMDWorkspace(1, 1; beta=1.0, R=1.0, norm=true)

    weights0 = [1.0]
    coords0 = reshape([0.0], 1, 1)
    weights1 = [1.0]
    coords1 = reshape([3.0], 1, 1)

    val, status = EnergyFlow._emd_raw!(ws, weights0, coords0, weights1, coords1)
    test_log("  raw solve: status=$status val=$val")
    @test status == :optimal
    @test val ≈ 3.0 atol=1e-10

    val_alloc, status_alloc = EnergyFlow._emd_raw_alloc(weights0, coords0, weights1, coords1;
                                                        beta=1.0, R=1.0, norm=true)
    test_log("  raw solve: alloc status=$status_alloc val=$val_alloc")
    @test status_alloc == :optimal
    @test val_alloc ≈ val atol=1e-10
end

@testset "EMDSolver unbalanced masses" begin
    test_log("  unbalanced: exercising fictitious particle branches")
    ws = EMDWorkspace(1, 1; beta=1.0, R=1.0, norm=false)
    coords0 = reshape([0.0], 1, 1)
    coords1 = reshape([0.0], 1, 1)

    heavier_source, status_source = EnergyFlow._emd_raw!(ws, [2.0], coords0, [1.0], coords1)
    test_log("  unbalanced: heavier_source status=$status_source val=$heavier_source")
    @test status_source == :optimal
    @test heavier_source ≈ 1.0 atol=1e-10

    heavier_target, status_target = EnergyFlow._emd_raw!(ws, [1.0], coords0, [2.0], coords1)
    test_log("  unbalanced: heavier_target status=$status_target val=$heavier_target")
    @test status_target == :optimal
    @test heavier_target ≈ 1.0 atol=1e-10
end

@testset "EMDSolver tiny nonzero imbalances" begin
    val_ns = emd_ns64([1e-12 0.0], [2e-12 0.0]; beta=1.0, R=1.0, norm=false)
    val_ns32 = emd_ns32([Float32(1e-12) Float32(0.0)], [Float32(2e-12) Float32(0.0)];
                        beta=1.0, R=1.0, norm=false)
    val_sk = emd_sinkhorn([1e-12 0.0], [2e-12 0.0]; beta=1.0, R=1.0, norm=false,
                          epsilon=1e-6, max_iter_sinkhorn=10_000)

    @test val_ns ≈ 1e-12 atol=1e-20
    @test Float64(val_ns32) ≈ 1e-12 atol=1e-20
    @test val_sk ≈ 1e-12 atol=1e-20
end

@testset "EMDSolver gdim and wrappers" begin
    test_log("  gdim/wrappers: verifying 1D vs 2D unpacking and backends")
    ev0 = [1.0 0.0 10.0]
    ev1 = [1.0 0.0 -20.0]

    gdim1 = emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true, gdim=1)
    gdim2 = emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true, gdim=2)
    test_log("  gdim/wrappers: ns64 gdim=1 => $gdim1, gdim=2 => $gdim2")
    @test gdim1 ≈ 0.0 atol=1e-10
    @test gdim2 ≈ 30.0 atol=1e-10

    ws64 = EMDWorkspace(1, 1; beta=1.0, R=1.0, norm=true)
    val_ns64 = emd_ns64!(ws64, ev0, ev1; gdim=1)
    val_ot64 = emd_ot64!(ws64, ev0, ev1; gdim=1)
    test_log("  gdim/wrappers: ns64!=$val_ns64 ot64!=$val_ot64")
    @test val_ns64 ≈ 0.0 atol=1e-10
    @test val_ot64 ≈ 0.0 atol=1e-10

    val_ns64_alloc = emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true, gdim=1)
    val_ot64_alloc = emd_ot64(ev0, ev1; R=1.0, beta=1.0, norm=true, gdim=1)
    test_log("  gdim/wrappers: ns64 alloc=$val_ns64_alloc ot64 alloc=$val_ot64_alloc")
    @test val_ns64_alloc ≈ val_ns64 atol=1e-10
    @test val_ot64_alloc ≈ val_ot64 atol=1e-10

    ws32 = EMDWorkspace{Float32}(1, 1; beta=1.0, R=1.0, norm=true)
    val_ns32 = emd_ns32!(ws32, ev0, ev1; gdim=1)
    val_ot32 = emd_ot32!(ws32, ev0, ev1; gdim=1)
    test_log("  gdim/wrappers: ns32!=$val_ns32 ot32!=$val_ot32")
    @test val_ns32 isa Float32
    @test val_ot32 isa Float32
    @test Float64(val_ns32) ≈ 0.0 atol=1e-6
    @test Float64(val_ot32) ≈ 0.0 atol=1e-6
end

@testset "EMDSolver pairwise matrix reconstruction" begin
    test_log("  pairwise reconstruction: exercising symmetric=false branch")

    events = [
        (Float64[1.0], reshape(Float64[0.0], 1, 1)),
        (Float64[1.0], reshape(Float64[1.0], 1, 1)),
        (Float64[1.0], reshape(Float64[3.0], 1, 1)),
    ]

    # Trigger the internal branch that reconstructs a full symmetric matrix from flat results
    D = EnergyFlow._pairwise_emd_self(Float64, events;
                                      beta=1.0, R=1.0, norm=true,
                                      max_iter=10_000, symmetric=false)

    test_log("  pairwise reconstruction: matrix size=$(size(D))")
    test_log("  pairwise reconstruction: D[1,2]=$(D[1,2]) D[1,3]=$(D[1,3]) D[2,3]=$(D[2,3])")

    @test size(D) == (3, 3)
    @test D[1, 1] ≈ 0.0 atol=1e-12
    @test D[2, 2] ≈ 0.0 atol=1e-12
    @test D[3, 3] ≈ 0.0 atol=1e-12
    @test D[1, 2] ≈ D[2, 1] atol=1e-12
    @test D[1, 3] ≈ D[3, 1] atol=1e-12
    @test D[2, 3] ≈ D[3, 2] atol=1e-12

    # Expected EMDs for single-particle events in 1D are absolute coordinate differences
    @test D[1, 2] ≈ 1.0 atol=1e-12
    @test D[1, 3] ≈ 3.0 atol=1e-12
    @test D[2, 3] ≈ 2.0 atol=1e-12
end

@testset "EMDSolver status warnings and strict failure" begin
    trivial_val = emd_ns64([1.0 0.0], [1.0 0.0]; norm=true, n_iter_max=0, strict=false)
    @test trivial_val == 0.0

    ev0 = [0.6 0.0; 0.4 1.0]
    ev1 = [0.5 0.0; 0.5 1.0]

    @test_logs (:warn, r"status=:max_iter") begin
        val = emd_ns64(ev0, ev1; norm=true, n_iter_max=0, strict=false)
        @test isnan(val)
    end

    @test_throws ErrorException emd_ns64(ev0, ev1; norm=true, n_iter_max=0, strict=true)
end

test_log("\nAll EMDSolver tests completed!")