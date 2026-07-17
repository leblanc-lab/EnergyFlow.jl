using Test
using EnergyFlow
include("test_helpers.jl")

test_log("="^70)
test_log("PAIRWISE EMD TESTS")
test_log("="^70)

# Helper functions for test data
function create_simple_events(n::Int)
    events = Matrix{Float64}[]
    for i in 1:n
        # Event with 2 particles in 1D coordinate space
        weights = [0.5, 0.5]
        coords = reshape([0.0 + i*0.5, 1.0 + i*0.5], 2, 1)
        # Combine as matrix (particles x (1+dimensions))
        event = hcat(weights, coords)
        push!(events, event)
    end
    return events
end

function create_varied_events(n::Int)
    events = Matrix{Float64}[]
    for i in 1:n
        nparticles = 1 + mod(i, 3)  # 1, 2, or 3 particles per event
        weights = ones(nparticles) / nparticles
        coords = randn(nparticles, 2)
        event = hcat(weights, coords)
        push!(events, event)
    end
    return events
end

# Self-pairwise tests
@testset "emds_ns64 — self-pairwise basics" begin
    test_log("  ns64 self-pairwise: 3 events")
    events = create_simple_events(3)
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    result type: $(typeof(dists)), length: $(length(dists))")
    @test isa(dists, Vector{Float64})
    @test length(dists) == 3  # 3*(3-1)/2
    @test all(≥(0), dists)    # distances non-negative
    @test dists[1] > 0.0      # events should be distinct
end

@testset "emds_ns64! — self-pairwise in-place" begin
    test_log("  ns64! self-pairwise: pre-allocated vector")
    events = create_simple_events(4)
    
    # Pre-allocate results vector
    npairs = 4 * 3 ÷ 2  # 6 pairs
    results = Vector{Float64}(undef, npairs)
    
    ret = emds_ns64!(results, events; R=1.0, beta=1.0, norm=true)
    
    test_log("    result: $(results[1:3])")
    @test ret === results  # Should return the same vector
    @test length(results) == npairs
    @test all(≥(0), results)
    @test results[1] > 0.0
end

@testset "emds_ot64 — self-pairwise (arc mixing)" begin
    test_log("  ot64 self-pairwise with arc mixing")
    events = create_simple_events(3)
    
    dists = emds_ot64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    result length: $(length(dists))")
    @test isa(dists, Vector{Float64})
    @test length(dists) == 3
    @test all(≥(0), dists)
end

@testset "emds_ot64! — self-pairwise in-place (arc mixing)" begin
    test_log("  ot64! self-pairwise in-place with arc mixing")
    events = create_simple_events(3)
    
    npairs = 3 * 2 ÷ 2
    results = Vector{Float64}(undef, npairs)
    
    ret = emds_ot64!(results, events; R=1.0, beta=1.0, norm=true)
    
    @test ret === results
    @test length(results) == npairs
    @test all(≥(0), results)
end

@testset "emds_ns32 — self-pairwise Float32" begin
    test_log("  ns32 self-pairwise with Float32")
    events = create_simple_events(3)
    
    dists = emds_ns32(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    element type: $(eltype(dists))")
    @test isa(dists, Vector{Float32})
    @test length(dists) == 3
    @test all(≥(0), dists)
end

@testset "emds_ns32! — self-pairwise Float32 in-place" begin
    test_log("  ns32! self-pairwise Float32 in-place")
    events = create_simple_events(3)
    
    npairs = 3 * 2 ÷ 2
    results = Vector{Float32}(undef, npairs)
    
    ret = emds_ns32!(results, events; R=1.0, beta=1.0, norm=true)
    
    @test ret === results
    @test eltype(results) == Float32
    @test length(results) == npairs
end

@testset "emds_ot32 — self-pairwise Float32 (arc mixing)" begin
    test_log("  ot32 self-pairwise Float32 with arc mixing")
    events = create_simple_events(3)
    
    dists = emds_ot32(events; R=1.0, beta=1.0, norm=true)
    
    @test isa(dists, Vector{Float32})
    @test length(dists) == 3
    @test all(≥(0), dists)
end

@testset "emds_ot32! — self-pairwise Float32 in-place (arc mixing)" begin
    test_log("  ot32! self-pairwise Float32 in-place with arc mixing")
    events = create_simple_events(3)
    
    npairs = 3 * 2 ÷ 2
    results = Vector{Float32}(undef, npairs)
    
    ret = emds_ot32!(results, events; R=1.0, beta=1.0, norm=true)
    
    @test ret === results
    @test eltype(results) == Float32
end

# Cross-pairwise tests
@testset "emds_ns64 — cross-pairwise" begin
    test_log("  ns64 cross-pairwise: 3x4 matrix")
    events_a = create_simple_events(3)
    events_b = create_simple_events(4)
    
    dists = emds_ns64(events_a, events_b; R=1.0, beta=1.0, norm=true)
    
    test_log("    result shape: $(size(dists))")
    @test isa(dists, Matrix{Float64})
    @test size(dists) == (3, 4)
    @test all(≥(0), dists)
end

@testset "emds_ot64 — cross-pairwise (arc mixing)" begin
    test_log("  ot64 cross-pairwise with arc mixing")
    events_a = create_simple_events(2)
    events_b = create_simple_events(3)
    
    dists = emds_ot64(events_a, events_b; R=1.0, beta=1.0, norm=true)
    
    @test isa(dists, Matrix{Float64})
    @test size(dists) == (2, 3)
    @test all(≥(0), dists)
end

@testset "emds_ns32 — cross-pairwise Float32" begin
    test_log("  ns32 cross-pairwise Float32")
    events_a = create_simple_events(2)
    events_b = create_simple_events(3)
    
    dists = emds_ns32(events_a, events_b; R=1.0, beta=1.0, norm=true)
    
    @test isa(dists, Matrix{Float32})
    @test size(dists) == (2, 3)
    @test all(≥(0), dists)
end

@testset "emds_ot32 — cross-pairwise Float32 (arc mixing)" begin
    test_log("  ot32 cross-pairwise Float32 with arc mixing")
    events_a = create_simple_events(2)
    events_b = create_simple_events(3)
    
    dists = emds_ot32(events_a, events_b; R=1.0, beta=1.0, norm=true)
    
    @test isa(dists, Matrix{Float32})
    @test size(dists) == (2, 3)
    @test all(≥(0), dists)
end

# Consistency: Self-pairwise matches individual EMD calls
@testset "Self-pairwise consistency with individual emd calls (ns64)" begin
    test_log("  Check pairwise results against individual calls")
    events = create_simple_events(3)
    
    # Compute via pairwise
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    # Compute individually
    d01 = emd_ns64(events[1], events[2]; R=1.0, beta=1.0, norm=true)
    d02 = emd_ns64(events[1], events[3]; R=1.0, beta=1.0, norm=true)
    d12 = emd_ns64(events[2], events[3]; R=1.0, beta=1.0, norm=true)
    
    test_log("    pairwise: $(dists)")
    test_log("    individual: $d01, $d02, $d12")
    
    # SciPy pdist order: (0,1), (0,2), (1,2)
    @test dists[1] ≈ d01 atol=1e-12
    @test dists[2] ≈ d02 atol=1e-12
    @test dists[3] ≈ d12 atol=1e-12
end

@testset "Self-pairwise consistency with individual emd calls (ot64)" begin
    test_log("  OT backend should match individual calls too")
    events = create_simple_events(3)
    
    dists = emds_ot64(events; R=1.0, beta=1.0, norm=true)
    
    d01 = emd_ot64(events[1], events[2]; R=1.0, beta=1.0, norm=true)
    d02 = emd_ot64(events[1], events[3]; R=1.0, beta=1.0, norm=true)
    d12 = emd_ot64(events[2], events[3]; R=1.0, beta=1.0, norm=true)
    
    @test dists[1] ≈ d01 atol=1e-12
    @test dists[2] ≈ d02 atol=1e-12
    @test dists[3] ≈ d12 atol=1e-12
end

@testset "Self-pairwise consistency with individual emd calls (ns32)" begin
    test_log("  ns32 pairwise should match individual calls")
    events = create_simple_events(3)
    
    dists = emds_ns32(events; R=1.0, beta=1.0, norm=true)
    
    d01 = emd_ns32(events[1], events[2]; R=1.0, beta=1.0, norm=true)
    d02 = emd_ns32(events[1], events[3]; R=1.0, beta=1.0, norm=true)
    d12 = emd_ns32(events[2], events[3]; R=1.0, beta=1.0, norm=true)
    
    @test Float64(dists[1]) ≈ Float64(d01) atol=1e-6
    @test Float64(dists[2]) ≈ Float64(d02) atol=1e-6
    @test Float64(dists[3]) ≈ Float64(d12) atol=1e-6
end

@testset "Cross-pairwise consistency with individual emd calls (ns64)" begin
    test_log("  Cross-pairwise should match individual calls")
    events_a = create_simple_events(2)
    events_b = create_simple_events(3)
    
    dists = emds_ns64(events_a, events_b; R=1.0, beta=1.0, norm=true)
    
    # Verify each element
    for i in 1:2
        for j in 1:3
            d_individual = emd_ns64(events_a[i], events_b[j]; R=1.0, beta=1.0, norm=true)
            @test dists[i, j] ≈ d_individual atol=1e-12
        end
    end
    
    test_log("    All $(2*3) cross-pairs verified")
end

# Try different parameter values
@testset "Parameter variation: beta values (ns64)" begin
    test_log("  Try different beta values")
    events = create_simple_events(3)
    
    for beta in [0.5, 1.0, 2.0]
        dists = emds_ns64(events; beta=beta, R=1.0, norm=true)
        test_log("    beta=$beta: $(dists[1])")
        @test all(≥(0), dists)
    end
end

@testset "Parameter variation: R values (ns64)" begin
    test_log("  Try different R values")
    events = create_simple_events(3)
    
    for R in [0.5, 1.0, 2.0]
        dists = emds_ns64(events; beta=1.0, R=R, norm=true)
        test_log("    R=$R: $(dists[1])")
        @test all(≥(0), dists)
    end
end

@testset "Parameter variation: norm flag (ns64)" begin
    test_log("  norm=true vs norm=false")
    events = create_simple_events(3)
    
    dists_norm_true = emds_ns64(events; beta=1.0, R=1.0, norm=true)
    dists_norm_false = emds_ns64(events; beta=1.0, R=1.0, norm=false)
    
    test_log("    norm=true: $(dists_norm_true[1])")
    test_log("    norm=false: $(dists_norm_false[1])")
    
    # Both should be valid distances
    @test all(≥(0), dists_norm_true)
    @test all(≥(0), dists_norm_false)
end

# Make sure different backends roughly agree
@testset "Backend consistency: ns64 vs ot64 (self-pairwise)" begin
    test_log("  NS and OT should give similar results")
    events = create_simple_events(3)
    
    dists_ns = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    dists_ot = emds_ot64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    NS: $(dists_ns)")
    test_log("    OT: $(dists_ot)")
    
    # OT variant (arc mixing) may differ slightly from NS
    for i in 1:length(dists_ns)
        @test abs(dists_ns[i] - dists_ot[i]) < 0.1 * max(dists_ns[i], 1.0)
    end
end

@testset "Float32/Float64 consistency: ns64 vs ns32" begin
    test_log("  Float32 and Float64 should be close enough")
    events = create_simple_events(3)
    
    dists64 = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    dists32 = emds_ns32(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    Float64: $(dists64)")
    test_log("    Float32: $(Float64.(dists32))")
    
    for i in 1:length(dists64)
        @test Float64(dists32[i]) ≈ dists64[i] atol=1e-5 rtol=1e-4
    end
end

@testset "Float32/Float64 consistency: ot64 vs ot32" begin
    test_log("  OT Float32 and Float64 should match too")
    events = create_simple_events(3)
    
    dists64 = emds_ot64(events; R=1.0, beta=1.0, norm=true)
    dists32 = emds_ot32(events; R=1.0, beta=1.0, norm=true)
    
    for i in 1:length(dists64)
        @test Float64(dists32[i]) ≈ dists64[i] atol=1e-5 rtol=1e-4
    end
end

# Edge cases and boundary conditions
@testset "Edge case: identical events (self-pairwise)" begin
    test_log("  Identical events should be zero distance")
    events = [
        [0.5 0.0; 0.5 1.0],
        [0.5 0.0; 0.5 1.0],  # identical to event 0
    ]
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    distance(identical): $(dists[1])")
    @test dists[1] < 1e-10  # should be essentially zero
end

@testset "Edge case: single particle events" begin
    test_log("  Single-particle events work")
    events = [
        reshape([1.0, 0.0], 1, 2),
        reshape([1.0, 1.0], 1, 2),
        reshape([1.0, 2.0], 1, 2),
    ]
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    distances: $(dists)")
    @test length(dists) == 3
    @test all(≥(0), dists)
end

@testset "Edge case: Many events (n=10)" begin
    test_log("  Works with more events (10)")
    events = create_simple_events(10)
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    expected_pairs = 10 * 9 ÷ 2  # 45
    test_log("    computed $(length(dists)) distances for $(expected_pairs) pairs")
    @test length(dists) == expected_pairs
    @test all(≥(0), dists)
end

# Make sure pre-allocation works right
@testset "In-place: Correct pre-allocation size" begin
    test_log("  In-place works with right size buffer")
    events = create_simple_events(4)
    npairs = 4 * 3 ÷ 2  # 6
    
    results = Vector{Float64}(undef, npairs)
    ret = emds_ns64!(results, events; R=1.0, beta=1.0, norm=true)
    
    @test ret === results
    @test length(results) == npairs
    @test all(≥(0), results)
end

@testset "In-place: Too-small pre-allocation should fail" begin
    test_log("  Errors on undersized buffer")
    events = create_simple_events(4)
    npairs = 4 * 3 ÷ 2
    
    too_small = Vector{Float64}(undef, npairs - 1)
    
    @test_throws AssertionError emds_ns64!(too_small, events)
end

# Internal helper: flat index to pair mapping
@testset "_flat_to_pair mapping (internal)" begin
    test_log("  Flat index to pair conversion works")
    
    n = 4
    pairs = []
    for k in 1:(n*(n-1)÷2)
        i, j = EnergyFlow._flat_to_pair(k, n)
        push!(pairs, (i, j))
        test_log("    k=$k -> ($i, $j)")
        @test 1 ≤ i < j ≤ n  # i < j constraint
    end
    
    expected = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    @test pairs == expected
end

# Test with various coordinate dimensions
@testset "Multi-dimensional coordinates: 2D coords" begin
    test_log("  Works with 2D coordinates")
    # Events with 2D coordinates (gdim=2)
    events = [
        [0.5 0.0 0.0; 0.5 1.0 1.0],
        [0.5 0.0 0.5; 0.5 1.0 0.5],
    ]
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    test_log("    distance in 2D space: $(dists[1])")
    @test length(dists) == 1
    @test dists[1] > 0
end

@testset "Multi-dimensional coordinates: 3D coords" begin
    test_log("  Works with 3D coordinates")
    events = [
        [0.5 0.0 0.0 0.0; 0.5 1.0 1.0 1.0],
        [0.5 0.0 0.5 0.5; 0.5 1.0 0.5 0.5],
    ]
    
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)
    
    @test length(dists) == 1
    @test dists[1] > 0
end

@testset "_pairwise_parallel! — partition correctness" begin
    nthreads = Threads.nthreads()
    sizes = [0, 1, 2, 3, 7,
             64 * nthreads,           # forces bsize ≥ 2
             128 * nthreads + 13,     # bsize > 1 with a partial trailing block
             2560 * nthreads]         # forces bsize clamped at 64
    for npairs in sizes
        test_log("  _pairwise_parallel! npairs=$npairs (nthreads=$nthreads)")
        counter = Threads.Atomic{Int}(0)     # counts workspace creations
        owner   = zeros(Int, npairs)         # 1-based id of the task that saw each k
        make_ws() = Threads.atomic_add!(counter, 1) + 1   # ws is just its own id
        work!(ws, k) = (owner[k] = ws)

        EnergyFlow._pairwise_parallel!(npairs, make_ws, work!)

        ntasks = npairs == 0 ? 0 : clamp(nthreads, 1, npairs)
        # exactly one workspace created per spawned task (no per-item allocation)
        @test counter[] == ntasks

        npairs == 0 && continue

        # every index visited exactly once (init 0 ⇒ a remaining 0 is a gap)
        @test all(>(0), owner)
        @test length(unique(owner)) == ntasks   # each task did some work, no extras

        # block-cyclic layout: whole blocks owned by one task, repeating every ntasks
        bsize   = clamp(npairs ÷ (32 * ntasks), 1, 64)
        blockof(k) = (k - 1) ÷ bsize
        for k in 2:npairs                        # (a) one owner per block
            if blockof(k) == blockof(k - 1)
                @test owner[k] == owner[k - 1]
            end
        end
        nblocks = blockof(npairs) + 1            # (b) owners cycle with period ntasks
        for b in 0:(nblocks - 1 - ntasks)
            @test owner[b * bsize + 1] == owner[(b + ntasks) * bsize + 1]
        end
    end
end

@testset "Edge case: single-event self-pairwise (npairs == 0)" begin
    test_log("  Single event ⇒ zero pairs, must not error (regression)")
    events = create_simple_events(1)

    for f in (emds_ns64, emds_ot64, emds_ns32, emds_ot32)
        dists = f(events; R=1.0, beta=1.0, norm=true)
        @test length(dists) == 0
    end

    # in-place with a correctly-sized (zero-length) buffer
    results = Vector{Float64}(undef, 0)
    ret = emds_ns64!(results, events; R=1.0, beta=1.0, norm=true)
    @test ret === results
    @test length(ret) == 0
end

@testset "Large self-pairwise consistency (exercises block-cyclic bsize > 1)" begin
    n = 25                                       # 300 pairs ⇒ bsize > 1 on 1–2 threads
    events = create_varied_events(n)
    dists = emds_ns64(events; R=1.0, beta=1.0, norm=true)

    npairs = n * (n - 1) ÷ 2
    @test length(dists) == npairs

    k = 0                                         # scipy pdist / _flat_to_pair order
    for i in 1:(n - 1), j in (i + 1):n
        k += 1
        d = emd_ns64(events[i], events[j]; R=1.0, beta=1.0, norm=true)
        @test dists[k] ≈ d atol=1e-10
    end
    test_log("    verified all $npairs self-pairs against individual emd calls")
end

@testset "Large cross-pairwise consistency (exercises block-cyclic bsize > 1)" begin
    na, nb = 12, 13                               # 156 pairs
    events_a = create_varied_events(na)
    events_b = create_varied_events(nb)
    D = emds_ns64(events_a, events_b; R=1.0, beta=1.0, norm=true)

    @test size(D) == (na, nb)
    for i in 1:na, j in 1:nb
        d = emd_ns64(events_a[i], events_b[j]; R=1.0, beta=1.0, norm=true)
        @test D[i, j] ≈ d atol=1e-10
    end
    test_log("    verified all $(na * nb) cross-pairs against individual emd calls")
end

test_log("\nAll PairwiseEMD tests completed!")

