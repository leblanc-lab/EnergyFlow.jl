using Test
using EnergyFlow
include("test_helpers.jl")

test_log("="^70)
test_log("UTILS TESTS")
test_log("="^70)

# _unpack_event
@testset "_unpack_event — basic extraction" begin
    test_log("  basic 2-particle 1D event")
    ev = [0.3 0.0;
          0.7 1.0]

    weights, coords = EnergyFlow._unpack_event(ev, nothing)

    @test weights ≈ [0.3, 0.7]
    @test size(coords) == (1, 2)  # dim × n_particles
    @test coords[1, 1] ≈ 0.0
    @test coords[1, 2] ≈ 1.0
end

@testset "_unpack_event — 2D coordinates" begin
    test_log("  3-particle 2D event")
    ev = [0.2  1.0  2.0;
          0.5  3.0  4.0;
          0.3  5.0  6.0]

    weights, coords = EnergyFlow._unpack_event(ev, nothing)

    @test weights ≈ [0.2, 0.5, 0.3]
    @test size(coords) == (2, 3)  # 2 dims, 3 particles
    @test coords[:, 1] ≈ [1.0, 2.0]
    @test coords[:, 2] ≈ [3.0, 4.0]
    @test coords[:, 3] ≈ [5.0, 6.0]
end

@testset "_unpack_event — gdim truncation" begin
    test_log("  gdim=1 should drop extra coordinate columns")
    ev = [0.4  1.0  9.9  9.9;
          0.6  2.0  8.8  8.8]

    weights, coords = EnergyFlow._unpack_event(ev, 1)

    @test weights ≈ [0.4, 0.6]
    @test size(coords) == (1, 2)  # only first coord dim kept
    @test coords[1, 1] ≈ 1.0
    @test coords[1, 2] ≈ 2.0
end

@testset "_unpack_event — gdim=2 out of 3 columns" begin
    test_log("  gdim=2 keeps first 2 coordinate columns")
    ev = [0.5  1.0  2.0  99.0;
          0.5  3.0  4.0  99.0]

    weights, coords = EnergyFlow._unpack_event(ev, 2)

    @test size(coords) == (2, 2)
    @test coords[:, 1] ≈ [1.0, 2.0]
    @test coords[:, 2] ≈ [3.0, 4.0]
end

@testset "_unpack_event — Float32 conversion" begin
    test_log("  Float32 type dispatch")
    ev = Float64[0.5 0.0; 0.5 1.0]

    weights, coords = EnergyFlow._unpack_event(Float32, ev, nothing)

    @test eltype(weights) == Float32
    @test eltype(coords) == Float32
    @test weights ≈ Float32[0.5, 0.5]
end

@testset "_unpack_event — single particle" begin
    test_log("  single-particle event edge case")
    ev = reshape([1.0, 0.5], 1, 2)

    weights, coords = EnergyFlow._unpack_event(ev, nothing)

    @test length(weights) == 1
    @test weights[1] ≈ 1.0
    @test size(coords) == (1, 1)
    @test coords[1, 1] ≈ 0.5
end

@testset "_unpack_event — gdim exceeds columns errors" begin
    test_log("  gdim larger than actual dims should throw")
    ev = [0.5 1.0; 0.5 2.0]  # only 1 coordinate column

    @test_throws ErrorException EnergyFlow._unpack_event(ev, 2)
end

@testset "_unpack_event — default Float64 dispatch" begin
    test_log("  default dispatch (no type arg) gives Float64")
    ev = Float32[0.5 0.0; 0.5 1.0]

    weights, coords = EnergyFlow._unpack_event(ev, nothing)

    @test eltype(weights) == Float64
    @test eltype(coords) == Float64
end

# _emd_raw_alloc
@testset "_emd_raw_alloc — basic solve" begin
    test_log("  alloc wrapper: 1 particle each, distance 3")
    w0 = [1.0];  c0 = reshape([0.0], 1, 1)
    w1 = [1.0];  c1 = reshape([3.0], 1, 1)

    val, status = EnergyFlow._emd_raw_alloc(w0, c0, w1, c1; beta=1.0, R=1.0, norm=true)

    test_log("    val=$val status=$status")
    @test status == :optimal
    @test val ≈ 3.0 atol=1e-10
end

@testset "_emd_raw_alloc — identical events" begin
    test_log("  alloc wrapper: identical events should give zero")
    w0 = [0.5, 0.5];  c0 = [0.0 1.0]
    w1 = [0.5, 0.5];  c1 = [0.0 1.0]

    val, status = EnergyFlow._emd_raw_alloc(w0, c0, w1, c1; beta=1.0, R=1.0, norm=true)

    @test status == :optimal
    @test val ≈ 0.0 atol=1e-10
end

@testset "_emd_raw_alloc — type promotion" begin
    test_log("  mixed input types get promoted to Float64")
    w0 = Float32[1.0];  c0 = Float32[0.0;;]
    w1 = Float64[1.0];  c1 = Float64[2.0;;]

    val, status = EnergyFlow._emd_raw_alloc(w0, c0, w1, c1; beta=1.0, R=1.0, norm=true)

    @test status == :optimal
    @test val isa Float64
    @test val ≈ 2.0 atol=1e-10
end

test_log("\nAll Utils tests completed!")

