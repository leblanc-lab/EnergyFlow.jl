using Test
using EnergyFlow
include("test_helpers.jl")

test_log("="^70)
test_log("NETWORK SIMPLEX TESTS")
test_log("="^70)

@testset "NetworkSimplex constructor default type" begin
    ns = NetworkSimplexSolver(3, 4)
    test_log("  constructor: type=$(typeof(ns)) max_n0=$(ns.max_n0) max_n1=$(ns.max_n1)")
    @test ns isa NetworkSimplexSolver{Float64}
    @test ns.max_n0 == 3
    @test ns.max_n1 == 4
end

@testset "NetworkSimplex status branches" begin
    @testset "supply mismatch" begin
        ns = NetworkSimplexSolver{Float64}(2, 2)
        ns.costs[1:4] .= 1.0
        status = network_simplex!(ns, [1.0, 0.0], [0.4, 0.4])
        test_log("  supply mismatch: status=$status")
        @test status == :supply_mismatch
    end

    @testset "max_iter (serial)" begin
        ns = NetworkSimplexSolver{Float64}(2, 2)
        ns.costs[1] = 1.0
        ns.costs[2] = 3.0
        ns.costs[3] = 2.0
        ns.costs[4] = 4.0

        status = network_simplex!(ns, [0.6, 0.4], [0.5, 0.5]; max_iter=0)
        test_log("  max_iter serial: status=$status n_iters=$(ns.n_iters)")
        @test status == :max_iter
    end

    @testset "unbounded via adversarial signed supplies" begin
        ns = NetworkSimplexSolver{Float64}(1, 1)
        ns.costs[1] = -10.0
        status = network_simplex!(ns, [-1.0], [-1.0])
        test_log("  unbounded adversarial: status=$status")
        @test status == :unbounded
    end

    @testset "infeasible via adversarial signed supplies" begin
        ns = NetworkSimplexSolver{Float64}(1, 1)
        ns.costs[1] = 0.0
        status = network_simplex!(ns, [-1.0], [-1.0])
        test_log("  infeasible adversarial: status=$status")
        @test status == :infeasible
    end
end

@testset "Paper mode lifecycle and max_iter cleanup" begin
    ns = NetworkSimplexSolver{Float64}(2, 2)
    ns.pivot_mode = :paper
    ns.costs[1] = 1.0
    ns.costs[2] = 3.0
    ns.costs[3] = 2.0
    ns.costs[4] = 4.0

    status = network_simplex!(ns, [0.6, 0.4], [0.5, 0.5]; max_iter=0)
    test_log("  paper mode: nthreads=$(Threads.nthreads()) status=$status work_epoch=$(ns.work_epoch[]) done_count=$(ns.done_count[])")
    @test status == :max_iter
    if Threads.nthreads() > 1
        @test ns.work_epoch[] == -1
    else
        @test ns.work_epoch[] >= 0
    end
end

@testset "Parallel-threshold dispatch (full parallel finder)" begin
    ns = NetworkSimplexSolver{Float64}(2, 2)
    ns.pivot_mode = :serial
    ns.parallel_threshold = 1  # arc_num=4 >= 1 => pick _find_entering_arc_parallel!
    ns.costs[1] = 1.0
    ns.costs[2] = 3.0
    ns.costs[3] = 2.0
    ns.costs[4] = 4.0

    status = network_simplex!(ns, [0.6, 0.4], [0.5, 0.5]; max_iter=1000)
    test_log("  parallel-threshold dispatch: status=$status n_iters=$(ns.n_iters)")
    @test status == :optimal
end

@testset "Paper mode optimal cleanup" begin
    ns = NetworkSimplexSolver{Float64}(2, 2)
    ns.pivot_mode = :paper
    ns.costs[1] = 1.0
    ns.costs[2] = 3.0
    ns.costs[3] = 2.0
    ns.costs[4] = 4.0

    status = network_simplex!(ns, [0.6, 0.4], [0.5, 0.5]; max_iter=1000)
    test_log("  paper optimal cleanup: nthreads=$(Threads.nthreads()) status=$status work_epoch=$(ns.work_epoch[]) done_count=$(ns.done_count[])")
    @test status == :optimal
    if Threads.nthreads() > 1
        @test ns.work_epoch[] == -1
    end
end

@testset "Paper mode unbounded in main loop" begin
    # Adversarial signed supplies that trigger unboundedness after at least one
    # pivot iteration (n_iters > 0), exercising the delta>=typemax(V) branch.
    ns = NetworkSimplexSolver{Float64}(2, 1)
    ns.pivot_mode = :paper
    ns.costs[1] = -3.915877931934624
    ns.costs[2] = -5.6906301598549245

    source_weights = [-1.4673454953491227, 1.378981185299811]
    target_weights = [-0.08836431004931167]

    status = network_simplex!(ns, source_weights, target_weights; max_iter=200)
    test_log("  paper unbounded main-loop: status=$status n_iters=$(ns.n_iters) work_epoch=$(ns.work_epoch[])")

    @test status == :unbounded
    @test ns.n_iters > 0
    if Threads.nthreads() > 1
        @test ns.work_epoch[] == -1
    end
end

@testset "Paper entering-arc false full scan" begin
    ns = NetworkSimplexSolver{Float64}(1, 1)
    ns.costs[1] = 1.0
    @test network_simplex!(ns, [1.0], [1.0]) == :optimal

    ns.paper_block_size = 1
    ns.next_arc = 1
    ns.states[1] = EnergyFlow.STATE_TREE  # guarantees no entering arc

    nworkers = Threads.nthreads() - 1
    if nworkers > 0
        ns.work_epoch[] = 0
        ns.done_count[] = 0
        for w in 1:nworkers
            ns.worker_tasks[w] = Threads.@spawn EnergyFlow._paper_worker_loop!(ns, w + 1)
        end
    end

    found = EnergyFlow._find_entering_arc_paper!(ns)
    test_log("  paper entering false scan: found=$found next_arc=$(ns.next_arc) nthreads=$(Threads.nthreads())")
    @test !found
    @test ns.next_arc == 1

    if nworkers > 0
        Threads.atomic_xchg!(ns.work_epoch, -1)
        for i in 1:nworkers
            wait(ns.worker_tasks[i])
        end
    end
end

@testset "Internal entering-arc scans" begin
    ns = NetworkSimplexSolver{Float64}(1, 1)
    ns.costs[1] = 1.0
    @test network_simplex!(ns, [1.0], [1.0]) == :optimal
    test_log("  entering-arc setup: status=$(ns.status) n_iters=$(ns.n_iters)")

    # Case 1: all arcs marked tree => no entering arc
    ns.states[1] = EnergyFlow.STATE_TREE
    test_log("  entering-arc case1: state=STATE_TREE")
    @test !EnergyFlow._find_entering_arc_parallel!(ns)

    # Case 2: tiny negative reduced cost fails epsilon criterion
    ns.states[1] = EnergyFlow.STATE_LOWER
    ns.sources[1] = 1
    ns.targets[1] = 2
    ns.costs[1] = 0.0
    ns.pis[1] = 0.0
    ns.pis[2] = 1e-20
    ns.art_cost_scale = 1.0
    test_log("  entering-arc case2: tiny rc with pis=$(ns.pis[1]),$(ns.pis[2])")
    @test !EnergyFlow._find_entering_arc_parallel!(ns)

    # Case 3: strong negative reduced cost is accepted
    ns.pis[2] = 1.0
    test_log("  entering-arc case3: strong rc with pis=$(ns.pis[1]),$(ns.pis[2])")
    @test EnergyFlow._find_entering_arc_parallel!(ns)
    @test ns.in_arc == 1
end

@testset "Paper worker loop" begin
    ns = NetworkSimplexSolver{Float64}(1, 1)
    ns.costs[1] = 1.0
    @test network_simplex!(ns, [1.0], [1.0]) == :optimal
    test_log("  worker setup: status=$(ns.status)")

    ns.states[1] = EnergyFlow.STATE_LOWER
    ns.sources[1] = 1
    ns.targets[1] = 2
    ns.costs[1] = 0.0
    ns.pis[1] = 0.0
    ns.pis[2] = 1.0

    ns.thread_work_start[1] = 1
    ns.thread_work_end[1] = 1

    worker = Threads.@spawn EnergyFlow._paper_worker_loop!(ns, 1)

    Threads.atomic_add!(ns.done_count, 1)
    Threads.atomic_add!(ns.work_epoch, 1)

    while ns.done_count[] > 0
        GC.safepoint()
    end

    test_log("  worker result: best_arc=$(ns.thread_best_arc[1]) min_rc=$(ns.thread_min_rc[1])")

    @test ns.thread_best_arc[1] == 1
    @test ns.thread_min_rc[1] < 0.0

    Threads.atomic_xchg!(ns.work_epoch, -1)
    wait(worker)
end

@testset "Leaving arc and state flip branch" begin
    ns = NetworkSimplexSolver{Float64}(1, 1)
    ns.costs[1] = 1.0
    @test network_simplex!(ns, [1.0], [1.0]) == :optimal
    test_log("  state flip setup: initial state=$(ns.states[1])")

    ns.in_arc = 1
    ns.states[1] = EnergyFlow.STATE_UPPER

    EnergyFlow._find_join_node!(ns)
    _ = EnergyFlow._find_leaving_arc!(ns)

    EnergyFlow._change_flow!(ns, false)
    test_log("  state flip result: final state=$(ns.states[1])")
    @test ns.states[1] == EnergyFlow.STATE_LOWER
end

test_log("\nAll NetworkSimplex tests completed!")
