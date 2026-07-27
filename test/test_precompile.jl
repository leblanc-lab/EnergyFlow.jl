# Regression tests guarding the re-enabled precompilation (see src/Precompile.jl
# and the "INVARIANT" comment in the NetworkSimplexSolver constructor).
#
# The package is precompiled with a thread count (usually 1) that need not match
# the runtime thread count. These tests assert that per-thread solver state is
# sized by the RUNTIME thread count, which is what makes the paper-mode worker
# pool and parallel pivot safe when precompiled at 1 thread and run at many.

@testset "Precompile safety: nthreads sizing" begin
    nt = Threads.nthreads()

    # A solver constructed at runtime must be sized by the runtime thread count,
    # not by whatever thread count was active during precompilation.
    ns = NetworkSimplexSolver{Float64}(4, 4)
    @test length(ns.worker_tasks) == max(nt - 1, 0)
    @test length(ns.thread_work_start) == nt
    @test length(ns.thread_work_end) == nt
    # thread_min_rc / thread_best_arc are padded by _CACHE_PAD per thread.
    @test length(ns.thread_min_rc) == length(ns.thread_best_arc)
    @test length(ns.thread_min_rc) == EnergyFlow._CACHE_PAD * nt

    # Float32 solver too.
    ns32 = NetworkSimplexSolver{Float32}(4, 4)
    @test length(ns32.worker_tasks) == max(nt - 1, 0)
    @test length(ns32.thread_work_start) == nt

    # A workspace built through the public constructor threads the same sizing.
    ws = EMDWorkspace(3, 3; norm=true)
    @test length(ws.ns.worker_tasks) == max(nt - 1, 0)
end

@testset "Precompile safety: threaded pairwise correct after precompile" begin
    # The core scenario the re-enabled precompilation had to survive: the pairwise
    # @threads path and the parallel @spawn cost-fill must agree with a serial
    # oracle regardless of the precompile-time thread count.
    events = [
        [0.6 0.1 0.2; 0.4 -0.3 0.5],
        [0.5 0.2 -0.1; 0.5 0.4 0.3],
        reshape([1.0, 0.0, 0.0], 1, 3),
        [0.3 0.5 0.5; 0.9 -0.2 0.1],
    ]

    for nrm in (true, false)
        par = emds(events; backend=:ns64, R=1.0, beta=1.0, norm=nrm)
        oracle = Float64[]
        for i in 1:length(events), j in (i+1):length(events)
            push!(oracle, emd(events[i], events[j]; backend=:ns64, R=1.0, beta=1.0, norm=nrm))
        end
        @test par ≈ oracle atol=1e-12
    end

    # Parallel cost-fill (n0*n1 above the threshold) must match the serial fill.
    # The default parallel_threshold is 40k, so n must satisfy n^2 ≥ 40k.
    n = 200
    big0 = hcat(rand(n) .+ 0.1, randn(n, 2))
    big1 = hcat(rand(n) .+ 0.1, randn(n, 2))
    ws_par = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
    @test n * n >= ws_par.parallel_threshold   # guard: really is the parallel path
    ws_ser = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
    @test emd!(ws_par, big0, big1) ≈ emd!(ws_ser, big0, big1) atol=1e-12
end
