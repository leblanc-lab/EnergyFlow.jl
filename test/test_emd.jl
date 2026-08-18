using Test
using EnergyFlow
include("test_helpers.jl")
test_log("="^70)
test_log("EMD TESTS")
test_log("="^70)
@testset "EMD dispatch" begin
    test_log("  Running EMD dispatch tests")
    ev0 = reshape([1.0, 0.0], 1, 2)
    ev1 = reshape([1.0, 1.0], 1, 2)
    ev2 = reshape([1.0, 2.0], 1, 2)
    events = [ev0, ev1, ev2]
    ev0_plan = [2.0 0.0; 1.0 1.0]
    ev1_plan = [1.0 0.5]

    original_backend = get_backend()
    try
        @test set_backend(:ns64) == :ns64
        @test get_backend() == :ns64

        ns64_val = emd(ev0, ev1; backend=:ns64, R=1.0, beta=1.0, norm=true)
        ot64_val = emd(ev0, ev1; backend=:ot64, R=1.0, beta=1.0, norm=true)
        test_log("  ns64 = $ns64_val, ot64 = $ot64_val")
        @test ns64_val ≈ emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true)
        @test ot64_val ≈ emd_ot64(ev0, ev1; R=1.0, beta=1.0, norm=true)

        ns64_val_flow, ns64_plan = emd(ev0_plan, ev1_plan; backend=:ns64, R=1.0, beta=1.0, norm=true, return_flow=true)
        ot64_val_flow, ot64_plan = emd(ev0_plan, ev1_plan; backend=:ot64, R=1.0, beta=1.0, norm=true, return_flow=true)
        @test ns64_val_flow ≈ 0.5 atol=1e-10
        @test ot64_val_flow ≈ 0.5 atol=1e-10
        @test size(ns64_plan) == (2, 1)
        @test size(ot64_plan) == (2, 1)
        @test ns64_plan ≈ Float64[2/3; 1/3]
        @test ot64_plan ≈ Float64[2/3; 1/3]

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

        metric = CustomMetric((a, b) -> 2 * sqrt(sum((a .- b) .^ 2)))
        metric_val = emd!(ws64, ev0, ev1; backend=:ns64, metric=metric)
        @test metric_val ≈ emd(ev0, ev1; backend=:ns64, R=1.0, beta=1.0, norm=true, metric=metric)
        @test metric_val ≈ 2.0 atol=1e-10

        results_metric = zeros(Float64, 3)
        emds!(results_metric, events; backend=:ns64, R=1.0, beta=1.0, norm=true, metric=metric)
        @test all(isapprox.(results_metric, emds(events; backend=:ns64, R=1.0, beta=1.0, norm=true, metric=metric); atol=1e-10))

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

        cross_ns32 = emds(events, [ev0, ev2]; backend=:ns32, R=1.0, beta=1.0, norm=true)
        cross_ot32 = emds(events, [ev0, ev2]; backend=:ot32, R=1.0, beta=1.0, norm=true)
        test_log("  cross ns32 = $cross_ns32")
        test_log("  cross ot32 = $cross_ot32")
        @test size(cross_ns32) == (3, 2)
        @test size(cross_ot32) == (3, 2)

        cross_sink = emds(events, [ev0, ev2]; backend=:sinkhorn, R=1.0, beta=1.0, norm=true)
        test_log("  cross sinkhorn = $cross_sink")
        @test size(cross_sink) == (3, 2)
        @test all(isfinite, cross_sink)

        results64 = zeros(Float64, 3)
        emds!(results64, events; backend=:ns64, R=1.0, beta=1.0, norm=true)
        test_log("  emds! ns64 = $results64")
        @test all(isapprox.(results64, self_ns; atol=1e-10))

        results64_ot = zeros(Float64, 3)
        emds!(results64_ot, events; backend=:ot64, R=1.0, beta=1.0, norm=true)
        test_log("  emds! ot64 = $results64_ot")
        @test all(isapprox.(results64_ot, self_ot; atol=1e-10))

        results32 = zeros(Float32, 3)
        emds!(results32, events; backend=:ns32, R=1.0, beta=1.0, norm=true)
        test_log("  emds! ns32 = $results32")
        @test all(isapprox.(Float64.(results32), Float64.(self_ns); atol=1e-5))

        self_ot32 = emds(events; backend=:ot32, R=1.0, beta=1.0, norm=true)
        results32_ot = zeros(Float32, 3)
        emds!(results32_ot, events; backend=:ot32, R=1.0, beta=1.0, norm=true)
        test_log("  pairwise ot32 = $self_ot32")
        test_log("  emds! ot32 = $results32_ot")
        @test all(isapprox.(Float64.(results32_ot), Float64.(self_ot32); atol=1e-5))

        sink_val = emd(ev0, ev1; backend=:sinkhorn, R=1.0, beta=1.0, norm=true)
        test_log("  sinkhorn = $sink_val")
        @test isfinite(sink_val)
        @test sink_val ≥ 0.0
        @test_throws ErrorException emd(ev0, ev1; backend=:sinkhorn, R=1.0, beta=1.0, norm=true, metric=metric)

        sink_val_flow, sink_plan = emd(ev0_plan, ev1_plan; backend=:sinkhorn, R=1.0, beta=1.0, norm=true, return_flow=true)
        @test sink_val_flow ≈ 0.5 atol=1e-10
        @test size(sink_plan) == (2, 1)
        @test sink_plan ≈ Float64[2/3; 1/3]

        ns64_unbalanced_val, ns64_unbalanced_plan = emd([1.0 0.0], [0.5 1.0]; backend=:ns64, norm=false, return_flow=true)
        sink_unbalanced_val, sink_unbalanced_plan = emd([1.0 0.0], [0.5 1.0]; backend=:sinkhorn, norm=false, return_flow=true)
        @test ns64_unbalanced_val ≈ sink_unbalanced_val atol=1e-10
        @test size(ns64_unbalanced_plan) == (1, 2)
        @test size(sink_unbalanced_plan) == (1, 2)
        @test ns64_unbalanced_plan ≈ Float64[0.5 0.5]
        @test sink_unbalanced_plan ≈ Float64[0.5 0.5]

        ev_balanced = [50.0 0.0; 50.0 1.0]
        ns_balanced_val, ns_balanced_plan = emd(ev_balanced, ev_balanced; backend=:ns64, norm=false, return_flow=true)
        sink_balanced_val, sink_balanced_plan = emd(ev_balanced, ev_balanced; backend=:sinkhorn, norm=false, return_flow=true)
        @test ns_balanced_val ≈ 0.0 atol=1e-10
        @test sink_balanced_val ≥ 0.0
        @test sum(ns_balanced_plan) ≈ 100.0 atol=1e-10
        @test sum(sink_balanced_plan) ≈ 100.0 atol=1e-8

        sink_ws = SinkhornWorkspace(2, 2; beta=1.0, R=1.0, norm=true, epsilon=0.01)
        sink_inplace = emd!(sink_ws, ev0_plan, ev1_plan)
        test_log("  sinkhorn emd! = $sink_inplace")
        @test sink_inplace isa Float64
        @test_throws ErrorException emd!(sink_ws, ev0, ev1; metric=metric)

        sink_inplace_flow, sink_inplace_plan = emd!(sink_ws, ev0_plan, ev1_plan; return_flow=true)
        @test sink_inplace_flow ≈ sink_inplace atol=1e-10
        @test size(sink_inplace_plan) == (2, 1)

        @testset "empty-event handling" begin
            empty_ev = Matrix{Float64}(undef, 0, 2)
            nonempty_ev = reshape([1.0, 0.0], 1, 2)

            @test emd(empty_ev, empty_ev; backend=:ns64, norm=true) == 0.0
            @test emd(empty_ev, empty_ev; backend=:ns64, norm=false) == 0.0
            @test emd(empty_ev, nonempty_ev; backend=:ns64, norm=true) == 0.0
            @test emd(empty_ev, nonempty_ev; backend=:ns64, norm=false) ≈ 1.0 atol=1e-12
            @test emd(nonempty_ev, empty_ev; backend=:ns64, norm=true) == 0.0
            @test emd(nonempty_ev, empty_ev; backend=:ns64, norm=false) ≈ 1.0 atol=1e-12

            empty_events = Vector{Matrix{Float64}}()
            @test isempty(emds(empty_events; backend=:ns64))
            @test isempty(emds([nonempty_ev]; backend=:ns64))
            @test size(emds(empty_events, [nonempty_ev]; backend=:ns64)) == (0, 1)
            @test size(emds([nonempty_ev], empty_events; backend=:ns64)) == (1, 0)
            @test size(emds(empty_events, [nonempty_ev, nonempty_ev]; backend=:ns64)) == (0, 2)
            @test size(emds([nonempty_ev, nonempty_ev], empty_events; backend=:ns64)) == (2, 0)
        end

        @test_throws ErrorException emd(ev0, ev1; backend=:bogus)
        @test_throws ErrorException emd!(ws64, ev0, ev1; backend=:bogus)
        @test_throws ErrorException emds(events; backend=:bogus)
        @test_throws ErrorException emds!(results64, events; backend=:sinkhorn)
        @test_throws ErrorException emds!(results64, events; backend=:bogus)
    finally
        set_backend(original_backend)
    end
end

test_log("\nAll EMD tests completed!")