#=
    NetworkSimplex.jl

    Reference: 
    1. LEMON's network simplex implementation (https://lemon.cs.elte.hu/trac/lemon)
    2. Kara, Özturan, Parallel network simplex algorithm for the minimum cost flow
       (the basis for the :parallel_block pivot mode)

    - Block-search pivot with epsilon-scaled stopping criterion
    - succ_num-based LCA (not depth-based)
    - forwards[] flag per node for arc direction
    - Pre-stored sources/targets for ALL arcs (bipartite + artificial)
    - Incremental thread/succ_num/last_succ update — NO full DFS rebuild
    - Heuristic initialPivots for each target node
    - Potential convention: rc = state * (cost + pi[src] - pi[tgt])

    Graph layout (1-indexed):
    - Nodes 1..n0      = sources
    - Nodes n0+1..n0+n1 = targets
    - Node  n0+n1+1    = artificial root
    - Arcs  1..n0*n1   = bipartite arcs (arc a: source=(a-1)÷n1+1, target=n0+(a-1)%n1+1)
    - Arcs  arc_num+1..all_arc_num = artificial arcs to/from root
=#

# ── Arc state constants ───────────────────────────────────────────────────────
const STATE_TREE  = Int8(0)
const STATE_LOWER = Int8(1)
const STATE_UPPER = Int8(-1)

# ── Block-search parameters (matching C++) ────────────────────────────────────
const BLOCK_SIZE_FACTOR = 1.0
const MIN_BLOCK_SIZE    = 10
const _CACHE_PAD        = 8    # elements per cache line (64 bytes / 8 bytes)

"""
    NetworkSimplexSolver{V<:AbstractFloat}
    NetworkSimplexSolver(max_n0, max_n1)

Pre-allocated workspace for the network simplex algorithm on a complete
bipartite transportation graph with up to `max_n0` sources and `max_n1`
targets. All arrays are allocated once for the maximum problem size; the
solver itself ([`network_simplex!`](@ref)) performs no hot-path allocation.

The unparameterized constructor defaults to `Float64`.

This is the low-level solver underlying the `:ns64`/`:ot64`/`:ns32`/`:ot32`
backends; most users should use [`emd`](@ref)/[`emds`](@ref) instead. After a
solve, `total_cost`, `status`, and `n_iters` hold the result and diagnostics.

Based on [LEMON](https://lemon.cs.elte.hu)'s network simplex implementation,
with optional parallel entering-arc search following Kara & Özturan (2022).

The entering-arc rule is selected by the `pivot_mode` field:

- `:serial` (default) — single-threaded block search.
- `:parallel_block` — block search with the block split across a persistent
  worker pool (Kara & Özturan). The mode to use when threads are available.
- `:full_parallel` — every pivot scans all arcs in parallel and takes the
  globally most-negative reduced cost. Kept as a simple baseline for
  benchmarking; it is expected to be slower than `:parallel_block`.
"""
mutable struct NetworkSimplexSolver{V<:AbstractFloat}
    # ── Dimensions ────────────────────────────────────────────────────────────
    n0::Int; n1::Int
    node_num::Int     # n0 + n1 + 1  (includes root)
    arc_num::Int      # n0 * n1
    all_arc_num::Int  # arc_num + node_num - 1  (includes artificial arcs)
    max_n0::Int; max_n1::Int

    # ── Epsilon thresholds ────────────────────────────────────────────────────
    epsilon_large::V   # supply-mismatch tolerance  = 1000 * eps(V)
    epsilon_small::V   # pivot optimality tolerance =    1 * eps(V)

    # ── Arc data (length = max_all_arc_num) ───────────────────────────────────
    costs::Vector{V}       # ground distances (caller fills [1..arc_num])
    flows::Vector{V}       # arc flows
    sources::Vector{Int}   # pre-stored source node of every arc
    targets::Vector{Int}   # pre-stored target node of every arc
    states::Vector{Int8}   # STATE_LOWER=1, STATE_TREE=0, STATE_UPPER=-1

    # ── Node data (length = max_node_num) ─────────────────────────────────────
    supply::Vector{V}   # supply (>0 source, <0 target, 0 root)
    pis::Vector{V}      # potentials: rc = state*(cost + pi[src] - pi[tgt])

    # ── Spanning tree (length = max_node_num) ─────────────────────────────────
    parent::Vector{Int}      # parent node (root has 0)
    thread::Vector{Int}      # next node in pre-order thread
    rev_thread::Vector{Int}  # previous node in pre-order thread
    succ_num::Vector{Int}    # subtree size (self inclusive)
    last_succ::Vector{Int}   # last node in subtree (thread order)
    pred::Vector{Int}        # predecessor arc (connecting node to parent)
    forwards::Vector{Bool}   # true ↔ pred arc goes node → parent

    # ── Scratch for incremental update ────────────────────────────────────────
    dirty_revs::Vector{Int}
    dirty_revs_len::Int

    # ── Block-search state ────────────────────────────────────────────────────
    block_size::Int
    next_arc::Int   # 1-indexed starting arc for next block scan

    # ── Pivot working variables ───────────────────────────────────────────────
    in_arc::Int
    join_::Int
    u_in::Int; v_in::Int; u_out::Int; v_out::Int
    delta::V

    # ── Per-call scratch (initialPivots) ──────────────────────────────────────
    arc_mins::Vector{Int}   # length max_n1

    # ── Parallel pivot search ─────────────────────────────────────────────────
    thread_min_rc::Vector{V}     # per-thread minimum reduced cost (padded, stride = _CACHE_PAD)
    thread_best_arc::Vector{Int} # per-thread best arc index      (padded, stride = _CACHE_PAD)

    # ── Persistent worker pool for :parallel_block mode ───────────────────────
    # Workers are spawned ONCE per solve (not per block/pivot).
    # Coordination is via two Atomic{Int} counters — zero heap allocation per pivot.
    # Total allocations per solve ≈ (nthreads-1) × allocs_per_@spawn ≈ 50 (constant).
    thread_work_start::Vector{Int}    # block start assigned to each thread slot
    thread_work_end::Vector{Int}      # block end  assigned to each thread slot
    work_epoch::Threads.Atomic{Int}   # main increments to signal new block; workers spin on this
    done_count::Threads.Atomic{Int}   # each worker decrements on completion; main polls for 0
    worker_tasks::Vector{Task}        # Task handles for spawned workers (length = nthreads-1)

    # ── Pivot mode selection ──────────────────────────────────────────────────
    pivot_mode::Symbol  # :serial (block-search), :parallel_block, :full_parallel
    parallel_block_size::Int  # block size for :parallel_block = max(4*isqrt(arc_num), MIN_BLOCK_SIZE)

    # ── Result ────────────────────────────────────────────────────────────────
    total_cost::V
    status::Symbol
    n_iters::Int     # pivot count for the last solve (diagnostic)
    n_arc_scans::Int # total arc evaluations in _find_entering_arc! (diagnostic)

    # ── Stopping-criterion scale floor ───────────────────────────────────────
    # Prevents epsilon-scaled threshold from becoming too tight when potentials
    # are anchored near 0 (norm=false with imbalanced weights).
    art_cost_scale::V

    # ── Arc mixing (ot style strided interleave) ────────────────────────────
    arc_mixing::Bool

    function NetworkSimplexSolver{V}(max_n0::Int, max_n1::Int) where {V<:AbstractFloat}
        max_arc_num     = max_n0 * max_n1
        max_node_num    = max_n0 + max_n1 + 1
        max_all_arc_num = max_arc_num + max_node_num - 1
        # nt sizes the per-thread scratch and worker-pool arrays. It is
        # read at RUNTIME here (not constant-folded into precompiled code), so a
        # solver built at runtime always matches the runtime thread count.
        # INVARIANT: never construct a NetworkSimplexSolver/EMDWorkspace at module
        # load or inside a precompile workload and store it in a global — that would
        # freeze nt at the (usually 1) precompile-time thread count. Workspaces must
        # be built at runtime (they are: in the emd/emds frontends and pairwise tasks).
        nt = Threads.nthreads()
        new{V}(
            0, 0, 0, 0, 0, max_n0, max_n1,
            V(1000) * eps(V), V(1) * eps(V),
            # Arc arrays
            Vector{V}(undef, max_all_arc_num),
            Vector{V}(undef, max_all_arc_num),
            Vector{Int}(undef, max_all_arc_num),
            Vector{Int}(undef, max_all_arc_num),
            Vector{Int8}(undef, max_all_arc_num),
            # Node arrays
            Vector{V}(undef, max_node_num),
            Vector{V}(undef, max_node_num),
            # Tree arrays
            Vector{Int}(undef, max_node_num),
            Vector{Int}(undef, max_node_num),
            Vector{Int}(undef, max_node_num),
            Vector{Int}(undef, max_node_num),
            Vector{Int}(undef, max_node_num),
            Vector{Int}(undef, max_node_num),
            Vector{Bool}(undef, max_node_num),
            # dirty_revs
            Vector{Int}(undef, max_node_num), 0,
            # Block search
            MIN_BLOCK_SIZE, 1,
            # Pivot vars
            1, 0, 0, 0, 0, 0, zero(V),
            # arc_mins
            Vector{Int}(undef, max(max_n1, 1)),
            # Parallel scratch — used by :parallel_block and :full_parallel
            Vector{V}(undef, _CACHE_PAD * nt),      # thread_min_rc  (padded)
            Vector{Int}(undef, _CACHE_PAD * nt),    # thread_best_arc (padded)
            # Persistent worker pool fields
            Vector{Int}(undef, nt),
            Vector{Int}(undef, nt),
            Threads.Atomic{Int}(0),
            Threads.Atomic{Int}(0),
            Vector{Task}(undef, max(nt - 1, 0)),
            # Pivot mode
            :serial, MIN_BLOCK_SIZE,
            # result
            zero(V), :optimal, 0, 0,
            # art_cost_scale
            zero(V),
            # arc_mixing
            false,
        )
    end
end

NetworkSimplexSolver(max_n0::Int, max_n1::Int) = NetworkSimplexSolver{Float64}(max_n0, max_n1)

# ── Initialization ────────────────────────────────────────────────────────────

"""
    network_simplex!(ns, source_weights, target_weights; max_iter=100_000) -> Symbol

Solve the minimum-cost transportation problem between `source_weights` and
`target_weights` on the complete bipartite graph, using the pre-allocated
[`NetworkSimplexSolver`](@ref) `ns`.

Arc costs must be filled in `ns.costs[1:n0*n1]` beforehand, in row-major
order: the arc from source `i` to target `j` is at index `(i-1)*n1 + j`.
Source and target weights must sum to the same total (within tolerance).

Returns a status `Symbol`, one of `:optimal`, `:infeasible`, `:unbounded`,
`:supply_mismatch`, or `:max_iter`. On success the objective value is in
`ns.total_cost` and the optimal flows in `ns.flows[1:n0*n1]`.
"""
function network_simplex!(ns::NetworkSimplexSolver{V},
                          source_weights::AbstractVector{V},
                          target_weights::AbstractVector{V};
                          max_iter::Int = 100_000) where V

    n0 = length(source_weights)
    n1 = length(target_weights)

    @assert n0 <= ns.max_n0 && n1 <= ns.max_n1

    ns.n0          = n0
    ns.n1          = n1
    ns.node_num    = n0 + n1 + 1
    ns.arc_num     = n0 * n1
    ns.all_arc_num = ns.arc_num + ns.node_num - 1
    ns.block_size          = max(Int(BLOCK_SIZE_FACTOR * isqrt(ns.arc_num)), MIN_BLOCK_SIZE)
    ns.parallel_block_size = max(4 * isqrt(ns.arc_num), MIN_BLOCK_SIZE)
    ns.next_arc    = 1
    ns.in_arc      = 1

    root     = ns.node_num   # artificial root (1-indexed last node)
    arc_num  = ns.arc_num
    node_num = ns.node_num

    # ── Set supply values ─────────────────────────────────────────────────────
    @inbounds begin
        sum_s = zero(V)
        for i in 1:n0
            s = source_weights[i]
            ns.supply[i] = s
            sum_s += s
        end
        for j in 1:n1
            s = -target_weights[j]
            ns.supply[n0 + j] = s
            sum_s += s
        end
        ns.supply[root] = zero(V)

        if abs(sum_s) > ns.epsilon_large
            ns.status = :supply_mismatch
            return ns.status
        end
    end

    # ── Artificial cost ────────────────────────────────────────────────────────
    max_cost = zero(V)
    @inbounds for a in 1:arc_num
        c = ns.costs[a]
        if c > max_cost; max_cost = c; end
    end
    art_cost = (max_cost + one(V)) * V(n0 + n1)
    ns.art_cost_scale = art_cost   # floor for epsilon-scaled stopping criterion

    # ── Initialize bipartite arc states and flows ─────────────────────────────
    @inbounds for a in 1:arc_num
        ns.flows[a]  = zero(V)
        ns.states[a] = STATE_LOWER
    end

    # ── Pre-store bipartite arc sources/targets ───────────────────────────────
    @inbounds for a in 1:arc_num
        ns.sources[a] = (a - 1) ÷ n1 + 1
        ns.targets[a] = n0 + (a - 1) % n1 + 1
    end

    # ── Arc mixing (OT-style, the backend of ot64 ane 32) ────────────────────────────
    # Permutes bipartite arcs so that each block of size k samples all sources
    # rather than consecutive arcs from the same source. Speeds up
    # block-search pivot for degenerate problems (norm=false, imbalanced weights).
    if ns.arc_mixing
        k = ns.block_size
        arc_num_local = arc_num
        tmp_sources = ns.sources[1:arc_num_local]  # copy
        tmp_targets = ns.targets[1:arc_num_local]  # copy
        tmp_costs   = ns.costs[1:arc_num_local]    # copy

        i = 1
        j = 1
        @inbounds for a_orig in 1:arc_num_local
            ns.sources[i] = tmp_sources[a_orig]
            ns.targets[i] = tmp_targets[a_orig]
            ns.costs[i]   = tmp_costs[a_orig]
            i += k
            if i > arc_num_local
                j += 1
                i = j
            end
        end
    end

    # ── Initialize root node ──────────────────────────────────────────────────
    @inbounds begin
        ns.parent[root]     = 0
        ns.pred[root]       = 0
        ns.thread[root]     = 1
        ns.rev_thread[1]    = root
        ns.succ_num[root]   = node_num
        ns.last_succ[root]  = root - 1
        ns.pis[root]        = zero(V)
    end

    # ── Initialize artificial arcs and star tree ──────────────────────────────
    @inbounds for u in 1:(node_num - 1)
        art_arc = arc_num + u
        ns.states[art_arc]  = STATE_TREE
        ns.flows[art_arc]   = abs(ns.supply[u])
        ns.parent[u]        = root
        ns.pred[u]          = art_arc
        ns.thread[u]        = u + 1      # thread: 1→2→...→(node_num-1)→root
        ns.rev_thread[u+1]  = u          # (rev_thread[root] set above; rev_thread[1] set above for root→1)
        ns.succ_num[u]      = 1
        ns.last_succ[u]     = u

        if ns.supply[u] >= zero(V)
            # supply node: arc goes u → root (forwards = true)
            ns.forwards[u]  = true
            ns.pis[u]       = zero(V)
            ns.sources[art_arc] = u
            ns.targets[art_arc] = root
            ns.flows[art_arc]   = ns.supply[u]
            ns.costs[art_arc]   = zero(V)
        else
            # demand node: arc goes root → u (forwards = false)
            ns.forwards[u]  = false
            ns.pis[u]       = art_cost
            ns.sources[art_arc] = root
            ns.targets[art_arc] = u
            ns.flows[art_arc]   = -ns.supply[u]
            ns.costs[art_arc]   = art_cost
        end
    end
    # Fix thread wrap: last real node points to root
    @inbounds ns.thread[node_num - 1] = root
    # rev_thread[root] was set above (= 1 via root's thread = 1 sets rev_thread[1] = root)
    # But we need rev_thread[root] = node_num - 1:
    @inbounds ns.rev_thread[root] = node_num - 1

    # ── Heuristic initial pivots ───────────────────────────────────────────────
    if !_initial_pivots!(ns)
        ns.status = :unbounded
        return ns.status
    end

    # ── Spawn persistent workers for :parallel_block mode ─────────────────────
    # Workers are spawned here, once per solve.
    nworkers = 0
    if ns.pivot_mode == :parallel_block
        nworkers = Threads.nthreads() - 1
        ns.work_epoch[] = 0
        ns.done_count[] = 0
        for w in 1:nworkers
            ns.worker_tasks[w] = Threads.@spawn _parallel_block_worker_loop!(ns, w + 1)
        end
    end

    # ── Main simplex loop ─────────────────────────────────────────────────────
    n_iter = 0
    ns.n_iters = 0
    ns.n_arc_scans = 0
    pivot_fn = if ns.pivot_mode == :parallel_block
        _find_entering_arc_parallel_block!
    elseif ns.pivot_mode == :full_parallel
        _find_entering_arc_parallel!
    elseif ns.pivot_mode == :serial
        _find_entering_arc!
    else
        throw(ArgumentError("unknown pivot_mode $(ns.pivot_mode); expected " *
                            ":serial, :parallel_block or :full_parallel"))
    end
    @inbounds while pivot_fn(ns)
        n_iter += 1
        ns.n_iters += 1
        if n_iter > max_iter
            ns.status = :max_iter
            if nworkers > 0
                Threads.atomic_xchg!(ns.work_epoch, -1)
                for i in 1:nworkers; wait(ns.worker_tasks[i]); end
            end
            return ns.status
        end

        _find_join_node!(ns)
        change = _find_leaving_arc!(ns)

        if ns.delta >= typemax(V)
            ns.status = :unbounded
            if nworkers > 0
                Threads.atomic_xchg!(ns.work_epoch, -1)
                for i in 1:nworkers; wait(ns.worker_tasks[i]); end
            end
            return ns.status
        end

        _change_flow!(ns, change)

        if change
            _update_tree_structure!(ns)
            _update_potential!(ns)
        end
    end

    # ── Terminate persistent workers (:parallel_block mode) ───────────────────
    if nworkers > 0
        Threads.atomic_xchg!(ns.work_epoch, -1)   # negative epoch = termination signal
        for i in 1:nworkers
            wait(ns.worker_tasks[i])
        end
    end

    # ── Feasibility check ─────────────────────────────────────────────────────
    ns.status = :optimal
    @inbounds for u in 1:(node_num - 1)
        f = ns.flows[arc_num + u]
        if f != zero(V)
            if abs(f) > ns.epsilon_large
                ns.status = :infeasible
                break
            else
                ns.flows[arc_num + u] = zero(V)
            end
        end
    end

    # ── Compute total cost ────────────────────────────────────────────────────
    total = zero(V)
    @inbounds @simd for a in 1:arc_num
        total += ns.flows[a] * ns.costs[a]
    end
    ns.total_cost = total

    return ns.status
end

# ── Heuristic initial pivots (one per target node) ────────────────────────────

function _initial_pivots!(ns::NetworkSimplexSolver{V}) where V
    @inbounds begin
        # Find minimum-cost incoming arc for each target node
        for j in 1:ns.n1
            min_cost = typemax(V)
            min_arc  = 0
            a = j   # arcs to target j: j, j+n1, j+2*n1, ...
            while a <= ns.arc_num
                c = ns.costs[a]
                if c < min_cost
                    min_cost = c
                    min_arc  = a
                end
                a += ns.n1
            end
            ns.arc_mins[j] = min_arc
        end

        # Pivot each candidate arc if it has negative reduced cost
        for j in 1:ns.n1
            e = ns.arc_mins[j]
            e == 0 && continue
            c = ns.states[e] * (ns.costs[e] + ns.pis[ns.sources[e]] - ns.pis[ns.targets[e]])
            c >= zero(V) && continue

            ns.in_arc = e
            _find_join_node!(ns)
            change = _find_leaving_arc!(ns)
            ns.delta >= typemax(V) && return false
            _change_flow!(ns, change)
            if change
                _update_tree_structure!(ns)
                _update_potential!(ns)
            end
        end
    end
    return true
end

# ── Block-search pivot rule ────────────────────────────────

function _find_entering_arc!(ns::NetworkSimplexSolver{V}) where V
    arc_num        = ns.arc_num
    eps_small      = ns.epsilon_small
    art_cost_scale = ns.art_cost_scale   # floor: prevents threshold collapsing near 0

    min_rc = zero(V)
    cnt    = ns.block_size
    e      = ns.next_arc   # start at next_arc (1-indexed, may equal arc_num+1 → wraps)
    n_scanned = 0

    @inbounds for _ in 1:arc_num
        n_scanned += 1
        if e > arc_num; e = 1; end   # wrap ("if e==arcNum: e-=arcNum")

        c = ns.states[e] * (ns.costs[e] + ns.pis[ns.sources[e]] - ns.pis[ns.targets[e]])
        if c < min_rc
            min_rc    = c
            ns.in_arc = e
        end

        cnt -= 1
        if cnt == 0
            ia = ns.in_arc
            a  = max(abs(ns.pis[ns.sources[ia]]), abs(ns.pis[ns.targets[ia]]),
                     abs(ns.costs[ia]), art_cost_scale)
            if min_rc < -eps_small * a
                ns.next_arc = e
                ns.n_arc_scans += n_scanned
                return true
            end
            cnt = ns.block_size
        end

        e += 1
    end

    # Post-loop check (full scan completed)
    ia = ns.in_arc
    a  = max(abs(ns.pis[ns.sources[ia]]), abs(ns.pis[ns.targets[ia]]),
             abs(ns.costs[ia]), art_cost_scale)
    ns.n_arc_scans += n_scanned
    if min_rc < -eps_small * a
        ns.next_arc = e   # e may be arc_num+1; wraps on next call
        return true
    end
    return false
end

# ── Parallel full-scan pivot rule (:full_parallel) ────────────────────────────
#
# The simplest parallelization of the entering-arc search: every call scans ALL
# arcs (no block, no wrap-around state) and takes the globally most-negative
# reduced cost — i.e. a parallel Dantzig rule.
#
# Kept as a correctness/performance baseline for benchmarking against
# :parallel_block, which is the mode to use in production: :full_parallel
# touches every arc on every pivot and forks a fresh task group per pivot, so it
# is expected to be slower despite doing the same work per arc.
#
# `Threads.@threads` here is deliberately :dynamic (the default): the :static
# schedule cannot be nested inside another threaded region, and this solver is
# routinely called from worker tasks (e.g. the pairwise EMD drivers).

function _find_entering_arc_parallel!(ns::NetworkSimplexSolver{V}) where V
    arc_num   = ns.arc_num
    eps_small = ns.epsilon_small
    nthreads  = Threads.nthreads()
    chunk_size = max(1, (arc_num + nthreads - 1) ÷ nthreads)

    Threads.@threads :dynamic for t in 1:nthreads
        chunk_start = (t - 1) * chunk_size + 1
        chunk_end   = min(t * chunk_size, arc_num)

        local_min  = zero(V)
        local_best = 0

        if chunk_start <= arc_num
            @inbounds for e in chunk_start:chunk_end
                c = ns.states[e] * (ns.costs[e] + ns.pis[ns.sources[e]] - ns.pis[ns.targets[e]])
                if c < local_min
                    local_min  = c
                    local_best = e
                end
            end
        end

        ns.thread_min_rc[_CACHE_PAD*(t-1)+1]   = local_min
        ns.thread_best_arc[_CACHE_PAD*(t-1)+1] = local_best
    end

    ns.n_arc_scans += arc_num   # diagnostic: this rule always scans every arc

    # Reduce: find global minimum across threads
    global_min  = zero(V)
    global_best = 0
    @inbounds for t in 1:nthreads
        if ns.thread_min_rc[_CACHE_PAD*(t-1)+1] < global_min
            global_min  = ns.thread_min_rc[_CACHE_PAD*(t-1)+1]
            global_best = ns.thread_best_arc[_CACHE_PAD*(t-1)+1]
        end
    end

    global_best == 0 && return false

    # Apply same epsilon-scaled optimality criterion as serial version
    ia = global_best
    @inbounds a = max(abs(ns.pis[ns.sources[ia]]), abs(ns.pis[ns.targets[ia]]),
                      abs(ns.costs[ia]), ns.art_cost_scale)
    if global_min < -eps_small * a
        ns.in_arc = ia
        return true
    end
    return false
end

# ── Parallel block-search pivot rule (:parallel_block) ────────────────────────
#
# Kara & Özturan (2022): parallelize ONLY the entering-arc search.
# Key difference from the @threads-per-block approach: workers are spawned ONCE
# per solve in network_simplex!, not once per block or per pivot.
# Coordination uses two Threads.Atomic{Int} counters (zero heap allocation
# per pivot).  Total allocations per solve ≈ (nthreads-1) × allocs_per_spawn
# (a small constant, ≈50, independent of n or number of pivots).
#
# Memory model (matching the OpenMP shared-memory reduction):
#   - Shared arrays (read-only during scan): states, costs, pis, sources, targets
#   - Private per-thread accumulators (stack/pre-allocated): thread_min_rc[t],
#     thread_best_arc[t]  — fixed size, independent of block size or arc count
#   - Coordination: work_epoch (signal) and done_count (barrier) atomics
#
# Flow per block:
#   1. Main writes thread_work_start[t], thread_work_end[t] for each thread.
#   2. Main sets done_count += (nthreads-1), then increments work_epoch → workers wake.
#   3. Main scans its own slot (slot 1) concurrently with workers.
#   4. Workers finish → decrement done_count.  Main spin-waits for done_count == 0.
#   5. Sequential reduction over thread_min_rc/thread_best_arc.
#   6. If ε-criterion met, accept and return true.  Else next block.

# ── Worker task body (spawned once per solve by network_simplex!) ─────────────
#
# Each worker spins on work_epoch until the main thread increments it (new block).
# After scanning, it decrements done_count and returns to spinning.
# A negative epoch value is the termination signal.

function _parallel_block_worker_loop!(ns::NetworkSimplexSolver{V}, slot::Int) where V
    last_epoch = 0
    while true
        # Spin-wait for new work (epoch change) or termination (epoch < 0)
        epoch = ns.work_epoch[]
        while epoch == last_epoch
            GC.safepoint()
            epoch = ns.work_epoch[]
        end
        last_epoch = epoch
        epoch < 0 && break                   # termination signal

        # Scan assigned arc range [my_start, my_end]
        my_start = ns.thread_work_start[slot]
        my_end   = ns.thread_work_end[slot]
        local_min  = zero(V)
        local_best = 0
        @inbounds for a in my_start:my_end
            st = ns.states[a]
            if st != STATE_TREE
                rc = V(st) * (ns.costs[a] + ns.pis[ns.sources[a]] - ns.pis[ns.targets[a]])
                if rc < local_min
                    local_min  = rc
                    local_best = a
                end
            end
        end
        ns.thread_min_rc[_CACHE_PAD*(slot-1)+1]   = local_min
        ns.thread_best_arc[_CACHE_PAD*(slot-1)+1] = local_best

        Threads.atomic_sub!(ns.done_count, 1)   # signal this worker is done
    end
    return nothing
end

function _find_entering_arc_parallel_block!(ns::NetworkSimplexSolver{V}) where V
    arc_num    = ns.arc_num
    eps_small  = ns.epsilon_small
    nthreads   = Threads.nthreads()
    block_size = ns.parallel_block_size   # 4·√m, pre-computed in network_simplex!
    nworkers   = nthreads - 1          # number of pre-spawned worker tasks

    start         = ns.next_arc   # 1-indexed cyclic position
    total_scanned = 0

    while total_scanned < arc_num
        # Current block: [start, block_end], clipped at arc_num
        block_end = min(start + block_size - 1, arc_num)
        block_len = block_end - start + 1

        # ── Assign per-thread ranges ──────────────────────────────────────────
        chunk = (block_len + nthreads - 1) ÷ nthreads
        @inbounds for t in 1:nthreads
            ns.thread_work_start[t] = start + (t - 1) * chunk
            ns.thread_work_end[t]   = min(start + t * chunk - 1, block_end)
        end

        # ── Signal workers: set done_count then increment epoch ───────────────
        # done_count must be set BEFORE epoch increment so workers see it correctly.
        Threads.atomic_add!(ns.done_count, nworkers)
        Threads.atomic_add!(ns.work_epoch, 1)

        # ── Main thread scans slot 1 concurrently with workers ────────────────
        my_start = ns.thread_work_start[1]
        my_end   = ns.thread_work_end[1]
        local_min  = zero(V)
        local_best = 0
        @inbounds for a in my_start:my_end
            st = ns.states[a]
            if st != STATE_TREE
                rc = V(st) * (ns.costs[a] + ns.pis[ns.sources[a]] - ns.pis[ns.targets[a]])
                if rc < local_min
                    local_min  = rc
                    local_best = a
                end
            end
        end
        ns.thread_min_rc[1]   = local_min      # slot 1 → index 1 (padded)
        ns.thread_best_arc[1] = local_best

        # ── Wait for all workers to finish ──────────────────────────────────── 
        # This might drag down the performance for large N I think.
        while ns.done_count[] > 0
            GC.safepoint()
        end

        # ── Sequential reduction over thread-local results ────────────────────
        block_min = zero(V)
        best_arc  = 0
        @inbounds for t in 1:nthreads
            idx = _CACHE_PAD * (t - 1) + 1
            if ns.thread_min_rc[idx] < block_min
                block_min = ns.thread_min_rc[idx]
                best_arc  = ns.thread_best_arc[idx]
            end
        end

        # ── Accept if ε-scaled criterion met (matches serial version) ─────────
        if best_arc != 0
            ia = best_arc
            @inbounds a_scale = max(abs(ns.pis[ns.sources[ia]]),
                                    abs(ns.pis[ns.targets[ia]]),
                                    abs(ns.costs[ia]),
                                    ns.art_cost_scale)
            if block_min < -eps_small * a_scale
                ns.in_arc   = ia
                ns.next_arc = (block_end >= arc_num) ? 1 : block_end + 1
                return true
            end
        end

        total_scanned += block_len
        start = (block_end >= arc_num) ? 1 : block_end + 1
    end

    return false
end

# ── Find join node (LCA via succ_num) ────────────────────────────

function _find_join_node!(ns::NetworkSimplexSolver)
    @inbounds begin
        u = ns.sources[ns.in_arc]
        v = ns.targets[ns.in_arc]
        while u != v
            if ns.succ_num[u] < ns.succ_num[v]
                u = ns.parent[u]
            else
                v = ns.parent[v]
            end
        end
        ns.join_ = u
    end
    return nothing
end

# ── Find leaving arc (sets u_in/v_in at end) ────────────────────

function _find_leaving_arc!(ns::NetworkSimplexSolver{V}) where V
    @inbounds begin
        if ns.states[ns.in_arc] == STATE_LOWER
            first  = ns.sources[ns.in_arc]
            second = ns.targets[ns.in_arc]
        else
            first  = ns.targets[ns.in_arc]
            second = ns.sources[ns.in_arc]
        end

        ns.delta = V(Inf)
        result   = Int8(0)

        # Walk first → join (u-side)
        u = first
        while u != ns.join_
            d = ns.forwards[u] ? ns.flows[ns.pred[u]] : V(Inf)
            if d < ns.delta
                ns.delta  = d
                ns.u_out  = u
                result    = Int8(1)
            end
            u = ns.parent[u]
        end

        # Walk second → join (v-side); <= for tie-breaking toward v-side
        u = second
        while u != ns.join_
            d = ns.forwards[u] ? V(Inf) : ns.flows[ns.pred[u]]
            if d <= ns.delta
                ns.delta  = d
                ns.u_out  = u
                result    = Int8(2)
            end
            u = ns.parent[u]
        end

        if result == Int8(1)
            ns.u_in = first;  ns.v_in = second
        else
            ns.u_in = second; ns.v_in = first
        end

        return result != Int8(0)
    end
end

# ── Change flows along the cycle ───────────────────────────────

function _change_flow!(ns::NetworkSimplexSolver{V}, change::Bool) where V
    @inbounds begin
        if ns.delta > zero(V)
            val = V(ns.states[ns.in_arc]) * ns.delta
            ns.flows[ns.in_arc] += val

            u = ns.sources[ns.in_arc]
            while u != ns.join_
                ns.flows[ns.pred[u]] += ns.forwards[u] ? -val : val
                u = ns.parent[u]
            end

            u = ns.targets[ns.in_arc]
            while u != ns.join_
                ns.flows[ns.pred[u]] += ns.forwards[u] ? val : -val
                u = ns.parent[u]
            end
        end

        if change
            ns.states[ns.in_arc]            = STATE_TREE
            leaving_arc                      = ns.pred[ns.u_out]
            ns.states[leaving_arc]           = (ns.flows[leaving_arc] == zero(V)) ? STATE_LOWER : STATE_UPPER
        else
            ns.states[ns.in_arc] = -ns.states[ns.in_arc]
        end
    end
    return nothing
end

# ── Incremental tree structure update ──────────────────────────

function _update_tree_structure!(ns::NetworkSimplexSolver{V}) where V
    @inbounds begin
        u_in  = ns.u_in;  v_in  = ns.v_in
        u_out = ns.u_out

        u             = ns.last_succ[u_in]
        oldrev_thread = ns.rev_thread[u_out]
        old_succ_num  = ns.succ_num[u_out]
        old_last_succ = ns.last_succ[u_out]
        right         = ns.thread[u]
        stem          = u_in
        par_stem      = v_in

        ns.v_out = ns.parent[u_out]

        # Determine 'last': the node that follows u_out's subtree in the thread
        last = (oldrev_thread == v_in) ? ns.thread[ns.last_succ[u_out]] : ns.thread[v_in]

        # ── Thread manipulation along the stem path ───────────────────────────
        ns.thread[v_in]    = u_in
        ns.dirty_revs_len  = 1
        ns.dirty_revs[1]   = v_in

        while stem != u_out
            new_stem = ns.parent[stem]

            # Insert new_stem into thread after u (last_succ of stem)
            ns.thread[u]              = new_stem
            ns.dirty_revs_len        += 1
            ns.dirty_revs[ns.dirty_revs_len] = u

            # Remove stem's subtree from the thread
            w            = ns.rev_thread[stem]
            ns.thread[w] = right
            ns.rev_thread[right] = w

            # Reverse parent pointer for this stem node
            ns.parent[stem] = par_stem
            par_stem        = stem
            stem            = new_stem

            # Advance u (the "tail" of the current stem's subtree in the thread)
            u     = (ns.last_succ[stem] == ns.last_succ[par_stem]) ? ns.rev_thread[par_stem] : ns.last_succ[stem]
            right = ns.thread[u]
        end

        # Finalize u_out
        ns.parent[u_out]    = par_stem
        ns.last_succ[u_out] = u
        ns.thread[u]        = last
        ns.rev_thread[last] = u

        # Remove u_out's subtree from old position (unless oldrev_thread == v_in)
        if oldrev_thread != v_in
            ns.thread[oldrev_thread]     = right
            ns.rev_thread[right]         = oldrev_thread
        end

        # Fix rev_thread for all dirty nodes
        for k in 1:ns.dirty_revs_len
            du = ns.dirty_revs[k]
            ns.rev_thread[ns.thread[du]] = du
        end

        # ── Update pred, forwards, succ_num for stem nodes (u_out → u_in) ─────
        tmp_sc = 0
        tmp_ls = ns.last_succ[u_out]   # = u (just set above)
        u_cur  = u_out

        while u_cur != u_in
            w               = ns.parent[u_cur]
            ns.pred[u_cur]    = ns.pred[w]
            ns.forwards[u_cur] = !ns.forwards[w]
            tmp_sc           += ns.succ_num[u_cur] - ns.succ_num[w]
            ns.succ_num[u_cur] = tmp_sc
            ns.last_succ[w]   = tmp_ls
            u_cur             = w
        end

        ns.pred[u_in]     = ns.in_arc
        ns.forwards[u_in] = (u_in == ns.sources[ns.in_arc])
        ns.succ_num[u_in] = old_succ_num

        # ── Update last_succ toward the root ──────────────────────────────────
        # Determine propagation limits (0 = "no limit" in 1-indexed)
        up_limit_in  = 0
        up_limit_out = 0
        if ns.last_succ[ns.join_] == v_in
            up_limit_out = ns.join_
        else
            up_limit_in  = ns.join_
        end

        # Propagate last_succ from v_in toward root
        u_cur = v_in
        new_ls_in = ns.last_succ[u_out]
        while u_cur != up_limit_in && ns.last_succ[u_cur] == v_in
            ns.last_succ[u_cur] = new_ls_in
            u_cur = ns.parent[u_cur]
        end

        # Propagate last_succ from v_out toward root
        v_out  = ns.v_out
        new_ls = (ns.join_ != oldrev_thread && v_in != oldrev_thread) ? oldrev_thread : ns.last_succ[u_out]
        u_cur  = v_out
        while u_cur != up_limit_out && ns.last_succ[u_cur] == old_last_succ
            ns.last_succ[u_cur] = new_ls
            u_cur = ns.parent[u_cur]
        end

        # ── Update succ_num from v_in and v_out to join ───────────────────────
        u_cur = v_in
        while u_cur != ns.join_
            ns.succ_num[u_cur] += old_succ_num
            u_cur = ns.parent[u_cur]
        end

        u_cur = v_out
        while u_cur != ns.join_
            ns.succ_num[u_cur] -= old_succ_num
            u_cur = ns.parent[u_cur]
        end
    end
    return nothing
end

# ── Incremental potential update ─────────────────────────────────────────────
#
# Standard "smaller-side" optimisation: updating u_in's subtree by +sigma is
# equivalent to updating its complement by -sigma.  We always walk the smaller
# set, reducing cost from O(subtree) to O(min(subtree, node_num-subtree)).
# This is critical when a fictitious particle with large supply/demand creates a
# subtree of size ≈ node_num (as in norm=false with very unequal event weights).

function _update_potential!(ns::NetworkSimplexSolver{V}) where V
    @inbounds begin
        u_in  = ns.u_in
        sigma = ns.forwards[u_in] ?
            ns.pis[ns.v_in] - ns.pis[u_in] - ns.costs[ns.pred[u_in]] :
            ns.pis[ns.v_in] - ns.pis[u_in] + ns.costs[ns.pred[u_in]]

        iszero(sigma) && return nothing

        stop = ns.thread[ns.last_succ[u_in]]

        if ns.succ_num[u_in] * 2 <= ns.node_num
            # u_in's subtree is the smaller side — walk it and add sigma
            u = u_in
            while u != stop
                ns.pis[u] += sigma
                u = ns.thread[u]
            end
        else
            # Complement is smaller — walk it and subtract sigma (equivalent)
            u = stop
            while u != u_in
                ns.pis[u] -= sigma
                u = ns.thread[u]
            end
        end
    end
    return nothing
end
