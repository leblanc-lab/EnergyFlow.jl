using Test

# Add the package path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using EnergyFlow

const TEST_DEBUG = lowercase(get(ENV, "TEST_DEBUG", "false")) in ("1", "true", "yes", "on")
test_log(args...) = TEST_DEBUG ? println(args...) : nothing

@testset "EnergyFlow" begin
    include("test_distances.jl")
    include("test_emd.jl")
    include("test_emdsolver.jl")
    include("test_hepmc3.jl")
    include("test_sinkhorn.jl")

    @testset "Backend management" begin
        @test get_backend() == :ns64
        @test set_backend(:ns64) == :ns64
        @test_throws ErrorException set_backend(:nonexistent)
    end

    @testset "emd_ns64 — identical distributions" begin
        # EnergyFlow format: M×(1+gdim), column 1 = weights
        ev0 = [0.5 0.0; 0.5 1.0]  # 2 particles in 1D
        ev1 = [0.5 0.0; 0.5 1.0]

        val = emd_ns64(ev0, ev1; beta=1.0, R=1.0, norm=true)
        @test val ≈ 0.0 atol=1e-10
    end

    @testset "emd_ns64 — shifted distributions" begin
        ev0 = reshape([1.0, 0.0], 1, 2)  # 1 particle at x=0
        ev1 = reshape([1.0, 1.0], 1, 2)  # 1 particle at x=1

        val = emd_ns64(ev0, ev1; beta=1.0, R=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-10
    end

    @testset "emd_ns64 — beta=2" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 2.0], 1, 2)

        val = emd_ns64(ev0, ev1; beta=2.0, R=1.0, norm=true)
        @test val ≈ 4.0 atol=1e-10  # (2/1)^2 = 4
    end

    @testset "emd_ns64 — R scaling" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)

        val = emd_ns64(ev0, ev1; beta=1.0, R=2.0, norm=true)
        @test val ≈ 0.5 atol=1e-10  # (1/2)^1 = 0.5
    end

    @testset "emd_ns64! — workspace reuse" begin
        ws = EMDWorkspace(10, 10; beta=1.0, R=1.0, norm=true)

        ev0 = [0.5 0.0; 0.5 1.0]
        ev1 = [0.5 0.0; 0.5 1.0]

        val = emd_ns64!(ws, ev0, ev1)
        @test val ≈ 0.0 atol=1e-10

        # Reuse with different events
        ev2 = reshape([1.0, 0.0], 1, 2)
        ev3 = reshape([1.0, 1.0], 1, 2)
        val2 = emd_ns64!(ws, ev2, ev3)
        @test val2 ≈ 1.0 atol=1e-10
    end

    @testset "emd! — backend dispatch" begin
        ws = EMDWorkspace(10, 10; beta=1.0, R=1.0, norm=true)

        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)

        val = emd!(ws, ev0, ev1; backend=:ns64)
        @test val ≈ 1.0 atol=1e-10

        # Default backend
        val2 = emd!(ws, ev0, ev1)
        @test val2 ≈ 1.0 atol=1e-10
    end

    @testset "emd — allocating convenience" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)

        val = emd(ev0, ev1; R=1.0, beta=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-10
    end

    @testset "emds_ns64 — self-pairwise" begin
        events = [
            [0.5 0.0; 0.5 1.0],
            [0.5 0.0; 0.5 2.0],
            reshape([1.0, 0.5], 1, 2),
        ]

        dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
        @test length(dists) == 3  # 3*(3-1)/2
        @test dists[1] > 0.0  # d(1,2) should differ
    end

    @testset "emds — backend dispatch" begin
        events = [
            [0.5 0.0; 0.5 1.0],
            [0.5 0.0; 0.5 2.0],
            reshape([1.0, 0.5], 1, 2),
        ]

        dists = emds(events; R=1.0, beta=1.0, norm=true)
        @test length(dists) == 3
        @test dists[1] > 0.0
    end

    # ─── Float32 backends ─────────────────────────────────────────────

    @testset "emd_ns32 — identical distributions" begin
        ev0 = [0.5 0.0; 0.5 1.0]
        ev1 = [0.5 0.0; 0.5 1.0]
        val = emd_ns32(ev0, ev1; beta=1.0, R=1.0, norm=true)
        @test val ≈ 0.0 atol=1e-6
        @test val isa Float32
    end

    @testset "emd_ns32 — shifted distributions" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)
        val = emd_ns32(ev0, ev1; beta=1.0, R=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-6
        @test val isa Float32
    end

    @testset "emd_ot32 — shifted distributions" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)
        val = emd_ot32(ev0, ev1; beta=1.0, R=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-6
        @test val isa Float32
    end

    @testset "Float32 — beta and R variations" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 2.0], 1, 2)

        # beta=2
        val = emd_ns32(ev0, ev1; beta=2.0, R=1.0, norm=true)
        @test val ≈ 4.0f0 atol=1e-5

        # beta=0.5
        val = emd_ns32(ev0, ev1; beta=0.5, R=1.0, norm=true)
        @test val ≈ sqrt(2.0f0) atol=1e-5

        # R=2
        ev1b = reshape([1.0, 1.0], 1, 2)
        val = emd_ns32(ev0, ev1b; beta=1.0, R=2.0, norm=true)
        @test val ≈ 0.5f0 atol=1e-6

        # R=0.5
        val = emd_ns32(ev0, ev1b; beta=1.0, R=0.5, norm=true)
        @test val ≈ 2.0f0 atol=1e-5
    end

    @testset "ns32 vs ns64 cross-validation" begin
        ev0 = [0.3 0.0 0.0; 0.7 1.0 2.0]
        ev1 = [0.4 0.5 0.5; 0.6 1.5 1.0]

        for beta in [0.5, 1.0, 2.0], R in [0.5, 1.0, 2.0], nrm in [true, false]
            val64 = emd_ns64(ev0, ev1; beta=beta, R=R, norm=nrm)
            val32 = emd_ns32(ev0, ev1; beta=beta, R=R, norm=nrm)
            @test Float64(val32) ≈ val64 atol=1e-5 rtol=1e-4
        end
    end

    @testset "ot32 vs ot64 cross-validation" begin
        ev0 = [0.3 0.0 0.0; 0.7 1.0 2.0]
        ev1 = [0.4 0.5 0.5; 0.6 1.5 1.0]

        for beta in [0.5, 1.0, 2.0], R in [0.5, 1.0, 2.0], nrm in [true, false]
            val64 = emd_ot64(ev0, ev1; beta=beta, R=R, norm=nrm)
            val32 = emd_ot32(ev0, ev1; beta=beta, R=R, norm=nrm)
            @test Float64(val32) ≈ val64 atol=1e-5 rtol=1e-4
        end
    end

    @testset "emd_ns32! — workspace reuse" begin
        ws = EMDWorkspace{Float32}(10, 10; beta=1.0, R=1.0, norm=true)

        ev0 = [0.5 0.0; 0.5 1.0]
        ev1 = [0.5 0.0; 0.5 1.0]
        val = emd_ns32!(ws, ev0, ev1)
        @test val ≈ 0.0 atol=1e-6
        @test val isa Float32

        ev2 = reshape([1.0, 0.0], 1, 2)
        ev3 = reshape([1.0, 1.0], 1, 2)
        val2 = emd_ns32!(ws, ev2, ev3)
        @test val2 ≈ 1.0 atol=1e-6
    end

    @testset "Backend switching — ns32/ot32" begin
        ev0 = reshape([1.0, 0.0], 1, 2)
        ev1 = reshape([1.0, 1.0], 1, 2)

        set_backend(:ns32)
        @test get_backend() == :ns32
        val = emd(ev0, ev1; R=1.0, beta=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-6
        @test val isa Float32

        set_backend(:ot32)
        @test get_backend() == :ot32
        val = emd(ev0, ev1; R=1.0, beta=1.0, norm=true)
        @test val ≈ 1.0 atol=1e-6
        @test val isa Float32

        # Restore default
        set_backend(:ns64)
    end

    @testset "emds_ns32 — self-pairwise" begin
        events = [
            [0.5 0.0; 0.5 1.0],
            [0.5 0.0; 0.5 2.0],
            reshape([1.0, 0.5], 1, 2),
        ]

        dists = emds_ns32(events; R=1.0, beta=1.0, norm=true)
        @test length(dists) == 3
        @test dists[1] > 0.0
        @test eltype(dists) == Float32
    end

    @testset "emds_ot32 — self-pairwise" begin
        events = [
            [0.5 0.0; 0.5 1.0],
            [0.5 0.0; 0.5 2.0],
            reshape([1.0, 0.5], 1, 2),
        ]

        dists = emds_ot32(events; R=1.0, beta=1.0, norm=true)
        @test length(dists) == 3
        @test dists[1] > 0.0
        @test eltype(dists) == Float32
    end

    @testset "NetworkSimplex direct usage" begin
        # 2 sources, 2 targets
        ns = NetworkSimplexSolver{Float64}(2, 2)

        ns.costs[1] = 1.0
        ns.costs[2] = 2.0
        ns.costs[3] = 3.0
        ns.costs[4] = 4.0

        sw = [0.5, 0.5]
        tw = [0.5, 0.5]

        status = network_simplex!(ns, sw, tw)
        @test status == :optimal
        @test ns.total_cost ≈ 2.5 atol=1e-10
    end
end
