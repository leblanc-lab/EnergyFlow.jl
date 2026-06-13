using Test
using EnergyFlow
test_log("="^70)
test_log("EMD TESTS")
test_log("="^70)
@testset "EMD dispatch" begin
    test_log("  Running EMD dispatch tests")
    ev0 = reshape([1.0, 0.0], 1, 2)
    ev1 = reshape([1.0, 1.0], 1, 2)
    ev2 = reshape([1.0, 2.0], 1, 2)
    events = [ev0, ev1, ev2]

    original_backend = get_backend()
    try
        @test set_backend(:ns64) == :ns64
        @test get_backend() == :ns64

        ns64_val = emd(ev0, ev1; backend=:ns64, R=1.0, beta=1.0, norm=true)
        ot64_val = emd(ev0, ev1; backend=:ot64, R=1.0, beta=1.0, norm=true)
        test_log("  ns64 = $ns64_val, ot64 = $ot64_val")
        @test ns64_val ≈ emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true)
        @test ot64_val ≈ emd_ot64(ev0, ev1; R=1.0, beta=1.0, norm=true)

        val32 = emd(ev0, ev1; backend=:ns32, R=1.0, beta=1.0, norm=true)
        test_log("  ns32 = $val32")
        @test val32 isa Float32
        @test Float64(val32) ≈ 1.0 atol=1e-6

        val32_ot = emd(ev0, ev1; backend=:ot32, R=1.0, beta=1.0, norm=true)
        test_log("  ot32 = $val32_ot")
        @test val32_ot isa Float32
        @test Float64(val32_ot) ≈ 1.0 atol=1e-6

        ws64 = EMDWorkspace(2, 2; beta=1.0, R=1.0, norm=true)
        ns64_inplace = emd!(ws64, ev0, ev1; backend=:ns64)
        ot64_inplace = emd!(ws64, ev0, ev1; backend=:ot64)
        test_log("  emd! ns64 = $ns64_inplace, ot64 = $ot64_inplace")
        @test ns64_inplace ≈ 1.0 atol=1e-10
        @test ot64_inplace ≈ 1.0 atol=1e-10

        ws32 = EMDWorkspace{Float32}(2, 2; beta=1.0, R=1.0, norm=true)
        ns32_inplace = emd!(ws32, ev0, ev1; backend=:ns32)
        ot32_inplace = emd!(ws32, ev0, ev1; backend=:ot32)
        test_log("  emd! ns32 = $ns32_inplace, ot32 = $ot32_inplace")
        @test ns32_inplace isa Float32
        @test ot32_inplace isa Float32

        self_ns = emds(events; backend=:ns64, R=1.0, beta=1.0, norm=true)
        self_ot = emds(events; backend=:ot64, R=1.0, beta=1.0, norm=true)
        test_log("  pairwise ns64 = $self_ns")
        test_log("  pairwise ot64 = $self_ot")
        @test length(self_ns) == 3
        @test length(self_ot) == 3
        @test self_ns[1] ≈ 1.0 atol=1e-10
        @test self_ot[1] ≈ 1.0 atol=1e-10

        cross_ns = emds(events, [ev0, ev2]; backend=:ns64, R=1.0, beta=1.0, norm=true)
        test_log("  cross ns64 = $cross_ns")
        @test size(cross_ns) == (3, 2)
        @test cross_ns[1, 1] ≈ 0.0 atol=1e-10
        @test cross_ns[2, 2] ≈ 1.0 atol=1e-10

        results64 = zeros(Float64, 3)
        emds!(results64, events; backend=:ns64, R=1.0, beta=1.0, norm=true)
        test_log("  emds! ns64 = $results64")
        @test all(isapprox.(results64, self_ns; atol=1e-10))

        results32 = zeros(Float32, 3)
        emds!(results32, events; backend=:ns32, R=1.0, beta=1.0, norm=true)
        test_log("  emds! ns32 = $results32")
        @test all(isapprox.(Float64.(results32), Float64.(self_ns); atol=1e-5))

        sink_val = emd(ev0, ev1; backend=:sinkhorn, R=1.0, beta=1.0, norm=true)
        test_log("  sinkhorn = $sink_val")
        @test isfinite(sink_val)
        @test sink_val ≥ 0.0

        sink_ws = SinkhornWorkspace(2, 2; beta=1.0, R=1.0, norm=true, epsilon=0.01)
        sink_inplace = emd!(sink_ws, ev0, ev1)
        test_log("  sinkhorn emd! = $sink_inplace")
        @test sink_inplace isa Float64

        @test_throws ErrorException emd(ev0, ev1; backend=:bogus)
        @test_throws ErrorException emd!(ws64, ev0, ev1; backend=:bogus)
        @test_throws ErrorException emds(events; backend=:bogus)
        @test_throws ErrorException emds!(results64, events; backend=:sinkhorn)
    finally
        set_backend(original_backend)
    end
end

test_log("\nAll EMD tests completed!")