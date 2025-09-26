# Consolidated Network Simplex Solver for EMD computation
# This file contains all necessary components for the fast network simplex implementation
# Performance: Unnormalized 10x90: ~35-45s, Normalized 10x90: ~15-20s

using LinearAlgebra
using Statistics
using LoopVectorization
using StaticArrays
using Base.Threads

# Export main user-facing functions
export emd_network_simplex, emds_network_simplex


# ========== Network Simplex Core Implementation ==========



using LoopVectorization

export NetworkSimplex, compute!, EMDStatus, Success, Empty, SupplyMismatch, Unbounded, MaxIterReached, Infeasible, NSf32, NSf64

@enum EMDStatus::Int8 begin
    Success = 0
    Empty = 1
    SupplyMismatch = 2
    Unbounded = 3
    MaxIterReached = 4
    Infeasible = 5
end

@enum ArcState::Int8 begin
    STATE_UPPER = -1
    STATE_TREE = 0
    STATE_LOWER = 1
end

const BLOCK_SIZE_FACTOR = 1.0
const MIN_BLOCK_SIZE = 10
const INVALID = -1
const INVALID_COST_VALUE = -1.0

mutable struct NetworkSimplex{V<:AbstractFloat, A<:Integer, N<:Integer, B<:Integer}
    # Parameters
    n_iter_max::Int
    n_iter::Int
    epsilon_large::V
    epsilon_small::V
    
    # Large constants
    MAX::V
    INF::V
    
    # Cost flow storage
    costs::Vector{V}
    flows::Vector{V}
    supplies::Vector{V}
    pis::Vector{V}
    sources::Vector{N}
    targets::Vector{N}
    
    # Spanning tree structure
    parents::Vector{N}
    threads::Vector{N}
    rev_threads::Vector{N}
    succ_nums::Vector{N}
    last_succs::Vector{N}
    dirty_revs::Vector{N}
    preds::Vector{A}
    arc_mins::Vector{A}
    forwards::Vector{B}
    states::Vector{ArcState}
    state_mult::Vector{Int8}  # Precomputed state multipliers (+1, 0, or -1)

    # Block search pivot rule variables
    next_arc::A
    block_size::N
    
    # Other variables
    sum_supplies::V
    total_cost::V
    
    # Temporary pivot iteration data
    in_arc::A
    join::N
    u_in::N
    v_in::N
    u_out::N
    v_out::N
    delta::V
    
    # Graph dimensions
    n0::N
    n1::N
    node_num::N
    arc_num::A
    
    function NetworkSimplex{V,A,N,B}() where {V<:AbstractFloat, A<:Integer, N<:Integer, B<:Integer}
        new{V,A,N,B}(
            100000,  # n_iter_max
            0,       # n_iter
            1000 * eps(V),  # epsilon_large
            eps(V),         # epsilon_small
            typemax(V),     # MAX
            isinf(typemax(V)) ? typemax(V) : typemax(V),  # INF
            Vector{V}(),
            Vector{V}(),
            Vector{V}(),
            Vector{V}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{N}(),
            Vector{A}(),
            Vector{A}(),
            Vector{B}(),
            Vector{ArcState}(),
            Vector{Int8}(),  # state_mult
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        )
    end
    
    function NetworkSimplex{V,A,N,B}(n_iter_max::Int, epsilon_large_factor::V, epsilon_small_factor::V) where {V,A,N,B}
        ns = NetworkSimplex{V,A,N,B}()
        set_params!(ns, n_iter_max, epsilon_large_factor, epsilon_small_factor)
        return ns
    end
end

# Optimized type aliases for common configurations
const NSf32 = NetworkSimplex{Float32, Int32, Int32, UInt8}
const NSf64 = NetworkSimplex{Float64, Int32, Int32, UInt8}

# Default constructor
NetworkSimplex(::Type{V}=Float64) where {V} = NetworkSimplex{V,Int32,Int32,UInt8}()

function set_params!(ns::NetworkSimplex{V}, n_iter_max::Int, epsilon_large_factor::V, epsilon_small_factor::V) where V
    ns.n_iter_max = n_iter_max
    ns.epsilon_large = epsilon_large_factor * eps(V)
    ns.epsilon_small = epsilon_small_factor * eps(V)
end

function construct_graph!(ns::NetworkSimplex{V,A,N}, n0::Int, n1::Int) where {V,A,N}
    ns.n0 = N(n0)
    ns.n1 = N(n1)
    ns.node_num = ns.n0 + ns.n1
    ns.arc_num = A(ns.n0) * A(ns.n1)
    
    if n0 + n1 > typemax(N)
        throw(OverflowError("Too many nodes for Node type"))
    end
    if n0 != 0 && ns.arc_num ÷ n0 != n1
        throw(OverflowError("Too many arcs for Arc type"))
    end
end

# Graph access functions - all inlined for hot paths
@inline nodeNum(ns::NetworkSimplex) = ns.node_num
@inline arcNum(ns::NetworkSimplex) = ns.arc_num
@inline nsource(ns::NetworkSimplex) = ns.n0
@inline ntarget(ns::NetworkSimplex) = ns.n1

# Get node from arc (C++ uses 0-based, keep it 0-based internally)
@inline source(ns::NetworkSimplex, arc::Integer) = arc ÷ ns.n1
@inline target(ns::NetworkSimplex, arc::Integer) = (arc % ns.n1) + ns.n0

function compute!(ns::NetworkSimplex{V,A,N}, n0::Int, n1::Int) where {V,A,N}
    construct_graph!(ns, n0, n1)
    status = run!(ns)

    if status == Success
        # Use regular optimized loop for total cost calculation
        ns.total_cost = zero(V)
        @fastmath @inbounds @simd for a in 1:arcNum(ns)
            ns.total_cost = muladd(ns.flows[a], ns.costs[a], ns.total_cost)  # Fused multiply-add
        end
    else
        ns.total_cost = INVALID_COST_VALUE
    end

    return status
end

function run!(ns::NetworkSimplex{V,A,N,B}) where {V,A,N,B}
    # Reset vectors sized according to number of nodes
    all_node_num = nodeNum(ns) + 1  # includes extra 1 for root node
    # Resize if any vector is too small (they should all be the same size, but check to be safe)
    if length(ns.supplies) < all_node_num || length(ns.parents) < all_node_num
        resize!(ns.supplies, all_node_num)
        resize!(ns.pis, all_node_num)
        resize!(ns.parents, all_node_num)
        resize!(ns.threads, all_node_num)
        resize!(ns.rev_threads, all_node_num)
        resize!(ns.succ_nums, all_node_num)
        resize!(ns.last_succs, all_node_num)
        resize!(ns.preds, all_node_num)
        resize!(ns.forwards, all_node_num)
    end

    # Reset vectors sized according to number of arcs
    all_arc_num = arcNum(ns) + nodeNum(ns)
    if length(ns.costs) < all_arc_num
        resize!(ns.costs, all_arc_num)
        resize!(ns.flows, all_arc_num)
        resize!(ns.sources, all_arc_num)
        resize!(ns.targets, all_arc_num)
        resize!(ns.states, all_arc_num)
        resize!(ns.state_mult, all_arc_num)
    end
    
    # Zero out flow with optimized loop
    flows_vec = ns.flows
    @turbo for i in 1:arcNum(ns)
        flows_vec[i] = 0.0
    end
    
    # Store arcs (C++ 0-based, but we store in Julia 1-based arrays)
    @inbounds @simd for a in 0:(arcNum(ns)-1)
        ns.sources[a+1] = source(ns, a)  # Store 0-based node indices
        ns.targets[a+1] = target(ns, a)  # Store 0-based node indices
    end

    # Arc reordering disabled - testing showed it degrades performance
    # The sorting overhead outweighs cache locality benefits
    # if arcNum(ns) > 1000  # Only for very large problems
    #     reorderArcsBySource!(ns)
    # end
    
    # Check for empty problem (rare case - put after common path)
    if nodeNum(ns) == 0
        return Empty
    end
    
    # Check supply total and make secondary supplies negative (FUSED LOOP)
    ns.sum_supplies = zero(V)
    @fastmath @inbounds for i in 0:(nodeNum(ns)-1)  # 0-based node indices
        if i < nsource(ns)
            ns.sum_supplies += ns.supplies[i+1]  # +1 for Julia array
        else
            ns.supplies[i+1] *= -1  # +1 for Julia array
            ns.sum_supplies = muladd(one(V), ns.supplies[i+1], ns.sum_supplies)  # Fused multiply-add
        end
    end
    
    if abs(ns.sum_supplies) > ns.epsilon_large
        println("sum_supplies: ", ns.sum_supplies)
        return SupplyMismatch
    end
    ns.sum_supplies = zero(V)
    
    # Initialize artificial cost
    artcosts = if V <: Integer
        typemax(V) ÷ 2 + 1
    else
        (maximum(view(ns.costs, 1:arcNum(ns))) + 1) * nodeNum(ns)
    end
    
    # Initialize arc maps with SIMD and precompute state multipliers
    @inbounds @simd for i in 1:arcNum(ns)
        ns.states[i] = STATE_LOWER
        ns.state_mult[i] = Int8(1)  # STATE_LOWER gives +1
    end
    
    # Set data for artificial root node (C++ uses nodeNum() as root index)
    root = nodeNum(ns)  # 0-based index for root
    ns.parents[root+1] = -1  # -1 indicates no parent
    ns.preds[root+1] = -1
    ns.threads[root+1] = 0  # 0-based
    ns.rev_threads[0+1] = root  # 0-based
    ns.succ_nums[root+1] = nodeNum(ns) + 1
    ns.last_succs[root+1] = root - 1
    ns.supplies[root+1] = -ns.sum_supplies
    ns.pis[root+1] = zero(V)
    
    # EQ supply constraints
    e = arcNum(ns)  # 0-based arc index
    for u in 0:(nodeNum(ns)-1)  # 0-based node indices
        ns.parents[u+1] = root
        ns.preds[u+1] = e
        ns.threads[u+1] = u + 1  # Next node in thread
        ns.rev_threads[u+1+1] = u  # u+1 is the thread value, so rev_threads at that position
        ns.succ_nums[u+1] = 1
        ns.last_succs[u+1] = u
        ns.states[e+1] = STATE_TREE
        ns.state_mult[e+1] = Int8(0)  # STATE_TREE gives 0
        
        if ns.supplies[u+1] >= 0
            ns.forwards[u+1] = 1
            ns.pis[u+1] = zero(V)
            ns.sources[e+1] = u
            ns.targets[e+1] = root
            ns.flows[e+1] = ns.supplies[u+1]
            ns.costs[e+1] = zero(V)
        else
            ns.forwards[u+1] = 0
            ns.pis[u+1] = artcosts
            ns.sources[e+1] = root
            ns.targets[e+1] = u
            ns.flows[e+1] = -ns.supplies[u+1]
            ns.costs[e+1] = artcosts
        end
        e += 1
    end
    
    # Initialize block search pivot rule
    ns.next_arc = 0  # 0-based
    ns.block_size = max(N(round(BLOCK_SIZE_FACTOR * sqrt(V(arcNum(ns))))), N(MIN_BLOCK_SIZE))
    
    # Perform heuristic initial pivots
    if !initialPivots!(ns)
        return Unbounded
    end
    
    # Execute Network Simplex algorithm
    ns.n_iter = 0
    while findEnteringArc!(ns)
        # Max iterations is rarely hit - check after increment
        ns.n_iter += 1
        if ns.n_iter >= ns.n_iter_max
            return MaxIterReached
        end

        findJoinNode!(ns)
        change = findLeavingArc!(ns)
        # Unbounded is very rare
        if ns.delta >= ns.MAX
            return Unbounded
        end
        changeFlow!(ns, change)
        # change is usually true (successful pivot)
        if change
            updateTreeStructure!(ns)
            updatePotential!(ns)
        end
    end
    
    # Check feasibility
    @inbounds for e in arcNum(ns):(all_arc_num-1)  # 0-based
        # Most flows are 0 at this point
        if ns.flows[e+1] != 0
            # Rarely happens - numerical issues
            if abs(ns.flows[e+1]) > ns.epsilon_large
                println("Bad flow: ", ns.flows[e+1])
                return Infeasible
            else
                ns.flows[e+1] = zero(V)
            end
        end
    end
    
    return Success
end

# Optimized price block scanning function - inlined for hot path
@inline function price_block!(ns::NetworkSimplex{V,A,N}, e_start::A, e_stop::A,
                     best_val::Ref{V}, best_e::Ref{A}) where {V,A,N}
    costs = ns.costs
    sources = ns.sources
    targets = ns.targets
    pis = ns.pis
    state_mult = ns.state_mult

    local_best = best_val[]
    local_e = A(-1)

    # Manual loop unrolling - process 2 arcs per iteration
    n = e_stop - e_start + 1
    n_blocks = n ÷ 2

    @inbounds for i in 1:n_blocks
        idx1 = 2*i - 1
        idx2 = 2*i

        e1 = e_start + idx1
        e2 = e_start + idx2

        # First arc
        u1 = sources[e1] + 1
        v1 = targets[e1] + 1
        r1 = state_mult[e1] * (costs[e1] + pis[u1] - pis[v1])

        # Second arc
        u2 = sources[e2] + 1
        v2 = targets[e2] + 1
        r2 = state_mult[e2] * (costs[e2] + pis[u2] - pis[v2])

        # Update minimum
        if r1 < local_best
            local_best = r1
            local_e = e_start + idx1 - 1
        end
        if r2 < local_best
            local_best = r2
            local_e = e_start + idx2 - 1
        end
    end

    # Handle remaining arc if n is odd
    @inbounds if n % 2 == 1
        idx = n
        e = e_start + idx
        u = sources[e] + 1
        v = targets[e] + 1
        r = state_mult[e] * (costs[e] + pis[u] - pis[v])
        if r < local_best
            local_best = r
            local_e = e_start + idx - 1
        end
    end

    if local_e >= 0
        best_val[] = local_best
        best_e[] = local_e
    end
    return nothing
end

function findEnteringArc!(ns::NetworkSimplex{V,A,N}) where {V,A,N}
    arc_num = arcNum(ns)
    e0 = ns.next_arc  # 0-based starting position

    # Initialize search
    best_val = Ref(zero(V))
    best_e = Ref(A(-1))

    # Scan block starting from e0
    # First range: from e0 to end of arc array
    if e0 < arc_num
        block_end = A(min(e0 + ns.block_size - 1, arc_num - 1))
        price_block!(ns, e0, block_end, best_val, best_e)

        # Second range: wraparound from 0 if we didn't scan full block
        scanned = block_end - e0 + 1
        if scanned < ns.block_size && e0 > 0
            wrap_end = A(min(ns.block_size - scanned - 1, e0 - 1))
            if wrap_end >= 0
                price_block!(ns, A(0), wrap_end, best_val, best_e)
            end
        end
    end

    # Update next_arc WITHOUT modulo - use conditional instead
    new_next = e0 + ns.block_size
    ns.next_arc = A(new_next >= arc_num ? new_next - arc_num : new_next)

    # If we found a candidate in this block, check epsilon and return
    if best_e[] >= 0
        ns.in_arc = best_e[]

        # Compute epsilon check ONLY for the winner
        @inbounds begin
            e_idx = ns.in_arc + 1
            u = ns.sources[e_idx] + 1
            v = ns.targets[e_idx] + 1
            a = max(abs(ns.pis[u]), abs(ns.pis[v]))
        end

        if best_val[] < -ns.epsilon_small * a
            return true
        end
    end

    # Continue searching remaining arcs if block didn't find entering arc
    # This is the fallback for when block search doesn't succeed
    remaining_blocks = div(arc_num - 1, ns.block_size)
    for _ in 1:remaining_blocks
        e0 = ns.next_arc

        # First range: from e0 to end
        if e0 < arc_num
            block_end = A(min(e0 + ns.block_size - 1, arc_num - 1))
            price_block!(ns, e0, block_end, best_val, best_e)

            # Second range: wraparound
            scanned = block_end - e0 + 1
            if scanned < ns.block_size && e0 > 0
                wrap_end = A(min(ns.block_size - scanned - 1, e0 - 1))
                if wrap_end >= 0
                    price_block!(ns, A(0), wrap_end, best_val, best_e)
                end
            end
        end

        # Update next_arc without modulo
        new_next = e0 + ns.block_size
        ns.next_arc = A(new_next >= arc_num ? new_next - arc_num : new_next)

        # Check if we found an entering arc
        if best_e[] >= 0
            ns.in_arc = best_e[]

            @inbounds begin
                e_idx = ns.in_arc + 1
                u = ns.sources[e_idx] + 1
                v = ns.targets[e_idx] + 1
                a = max(abs(ns.pis[u]), abs(ns.pis[v]))
            end

            if best_val[] < -ns.epsilon_small * a
                return true
            end
        end
    end

    return false
end

function initialPivots!(ns::NetworkSimplex{V,A,N}) where {V,A,N}
    # Find minimum cost incoming arc for each demand node
    # Reuse arc_mins vector without reallocation
    resize!(ns.arc_mins, 0)  # Clear without deallocating
    
    for v in nsource(ns):(nodeNum(ns)-1)  # 0-based node indices
        mincost = typemax(V)
        min_arc = INVALID
        
        # firstIn for node v (0-based)
        arc = if v < ns.n0
            INVALID
        else
            ns.arc_num + v - ns.node_num  # 0-based arc
        end
        
        while arc != INVALID
            c = ns.costs[arc+1]  # +1 for Julia array
            if c < mincost
                mincost = c
                min_arc = arc  # Store 0-based
            end
            # nextIn
            arc -= ns.n1
            if arc < 0
                arc = INVALID
            end
        end
        
        if min_arc != INVALID
            push!(ns.arc_mins, min_arc)  # Store 0-based
        end
    end
    
    # Perform heuristic initial pivots
    @inbounds for a in ns.arc_mins
        ns.in_arc = a  # 0-based
        if ns.state_mult[ns.in_arc+1] * (ns.costs[ns.in_arc+1] +
            ns.pis[ns.sources[ns.in_arc+1]+1] - ns.pis[ns.targets[ns.in_arc+1]+1]) >= 0
            continue
        end

        findJoinNode!(ns)
        change = findLeavingArc!(ns)
        if ns.delta >= ns.MAX
            return false
        end
        changeFlow!(ns, change)
        if change
            updateTreeStructure!(ns)
            updatePotential!(ns)
        end
    end
    return true
end

@inline function findJoinNode!(ns::NetworkSimplex)
    @inbounds begin
        u = ns.sources[ns.in_arc+1]  # Get 0-based node index
        v = ns.targets[ns.in_arc+1]  # Get 0-based node index

        while u != v
            # Branch prediction: usually both branches are equally likely
            if ns.succ_nums[u+1] < ns.succ_nums[v+1]
                u = ns.parents[u+1]  # parents stores 0-based indices (or -1)
            else
                v = ns.parents[v+1]  # parents stores 0-based indices (or -1)
            end
        end
        ns.join = u  # Store 0-based
    end
end

@inline function findLeavingArc!(ns::NetworkSimplex{V}) where V
    @inbounds begin
        # Initialize first and second nodes according to cycle direction
        # STATE_LOWER is more common in practice
        if ns.states[ns.in_arc+1] == STATE_LOWER
            first = ns.sources[ns.in_arc+1]  # 0-based
            second = ns.targets[ns.in_arc+1]  # 0-based
        else
            first = ns.targets[ns.in_arc+1]  # 0-based
            second = ns.sources[ns.in_arc+1]  # 0-based
        end

        ns.delta = ns.INF
        result = 0

        # Search cycle along path from first node to root
        u = first
        while u != ns.join
            d = ns.forwards[u+1] != 0 ? ns.flows[ns.preds[u+1]+1] : ns.INF
            if d < ns.delta
                ns.delta = d
                ns.u_out = u  # Store 0-based
                result = 1
            end
            u = ns.parents[u+1]  # Get next 0-based parent
        end

        # Search cycle along path from second node to root
        u = second
        while u != ns.join
            d = ns.forwards[u+1] != 0 ? ns.INF : ns.flows[ns.preds[u+1]+1]
            if d <= ns.delta
                ns.delta = d
                ns.u_out = u  # Store 0-based
                result = 2
            end
            u = ns.parents[u+1]  # Get next 0-based parent
        end

        if result == 1
            ns.u_in = first  # 0-based
            ns.v_in = second  # 0-based
        else
            ns.u_in = second  # 0-based
            ns.v_in = first  # 0-based
        end

        return result != 0
    end
end

@inline function changeFlow!(ns::NetworkSimplex{V}, change::Bool) where V
    @inbounds begin
        # Augment along the cycle (almost always delta > 0)
        if ns.delta > 0
            @fastmath val = ns.state_mult[ns.in_arc+1] * ns.delta
            ns.flows[ns.in_arc+1] += val

            u = ns.sources[ns.in_arc+1]  # 0-based
            @fastmath while u != ns.join
                ns.flows[ns.preds[u+1]+1] += ns.forwards[u+1] != 0 ? -val : val
                u = ns.parents[u+1]  # Get next 0-based parent
            end

            u = ns.targets[ns.in_arc+1]  # 0-based
            @fastmath while u != ns.join
                ns.flows[ns.preds[u+1]+1] += ns.forwards[u+1] != 0 ? val : -val
                u = ns.parents[u+1]  # Get next 0-based parent
            end
        end

        # Update state of entering and leaving arcs
        # change is usually true (successful pivot)
        if change
            ns.states[ns.in_arc+1] = STATE_TREE
            ns.state_mult[ns.in_arc+1] = Int8(0)  # STATE_TREE multiplier

            # Flow is rarely exactly 0
            leaving_arc = ns.preds[ns.u_out+1]+1
            if ns.flows[leaving_arc] == 0
                ns.states[leaving_arc] = STATE_LOWER
                ns.state_mult[leaving_arc] = Int8(1)
            else
                ns.states[leaving_arc] = STATE_UPPER
                ns.state_mult[leaving_arc] = Int8(-1)
            end
        else
            # Flip the state
            current_state = ns.states[ns.in_arc+1]
            if current_state == STATE_LOWER
                ns.states[ns.in_arc+1] = STATE_UPPER
                ns.state_mult[ns.in_arc+1] = Int8(-1)
            else
                ns.states[ns.in_arc+1] = STATE_LOWER
                ns.state_mult[ns.in_arc+1] = Int8(1)
            end
        end
    end
end

function updateTreeStructure!(ns::NetworkSimplex{V,A,N}) where {V,A,N}
    u = ns.last_succs[ns.u_in+1]  # 0-based
    oldrev_threads = ns.rev_threads[ns.u_out+1]  # 0-based
    oldsucc_nums = ns.succ_nums[ns.u_out+1]
    oldlast_succs = ns.last_succs[ns.u_out+1]  # 0-based
    right = ns.threads[u+1]  # 0-based
    stem = ns.u_in  # 0-based
    par_stem = ns.v_in  # 0-based
    ns.v_out = ns.parents[ns.u_out+1]  # 0-based (or -1)
    
    # Handle case when oldrev_threads equals v_in
    last = if oldrev_threads == ns.v_in
        ns.threads[ns.last_succs[ns.u_out+1]+1]  # 0-based
    else
        ns.threads[ns.v_in+1]  # 0-based
    end
    
    # Update threads and parents along stem nodes
    ns.threads[ns.v_in+1] = ns.u_in  # Store 0-based
    resize!(ns.dirty_revs, 0)  # Clear without deallocating
    push!(ns.dirty_revs, ns.v_in)  # Store 0-based

    @inbounds while stem != ns.u_out
        # Insert next stem node into thread list
        new_stem = ns.parents[stem+1]  # Get 0-based parent
        ns.threads[u+1] = new_stem  # Store 0-based
        push!(ns.dirty_revs, u)  # Store 0-based

        # Remove subtree of stem from thread list
        w = ns.rev_threads[stem+1]  # Get 0-based
        ns.threads[w+1] = right  # Store 0-based
        ns.rev_threads[right+1] = w  # Store 0-based

        # Change parent node and shift stem nodes
        ns.parents[stem+1] = par_stem  # Store 0-based
        par_stem = stem  # 0-based
        stem = new_stem  # 0-based

        # Update u and right
        u = ns.last_succs[stem+1] == ns.last_succs[par_stem+1] ?
            ns.rev_threads[par_stem+1] : ns.last_succs[stem+1]  # 0-based
        right = ns.threads[u+1]  # 0-based
    end
    
    ns.parents[ns.u_out+1] = par_stem  # Store 0-based
    ns.threads[u+1] = last  # Store 0-based
    ns.rev_threads[last+1] = u  # Store 0-based
    ns.last_succs[ns.u_out+1] = u  # Store 0-based
    
    # Remove subtree of u_out from thread list
    if oldrev_threads != ns.v_in
        ns.threads[oldrev_threads+1] = right  # Store 0-based
        ns.rev_threads[right+1] = oldrev_threads  # Store 0-based
    end
    
    # Update rev_threads using new threads values
    for u in ns.dirty_revs  # u is 0-based
        ns.rev_threads[ns.threads[u+1]+1] = u  # Store 0-based
    end
    
    # Update preds, forwards, last_succs and succ_nums for stem nodes
    tmp_sc = N(0)
    tmp_ls = ns.last_succs[ns.u_out+1]  # 0-based
    u = ns.u_out  # 0-based

    @inbounds while u != ns.u_in
        w = ns.parents[u+1]  # Get 0-based parent
        ns.preds[u+1] = ns.preds[w+1]
        ns.forwards[u+1] = 1 - ns.forwards[w+1]
        tmp_sc += ns.succ_nums[u+1] - ns.succ_nums[w+1]
        ns.succ_nums[u+1] = tmp_sc
        ns.last_succs[w+1] = tmp_ls  # Store 0-based
        u = w  # 0-based
    end

    ns.preds[ns.u_in+1] = ns.in_arc  # Store 0-based arc
    ns.forwards[ns.u_in+1] = (ns.u_in == ns.sources[ns.in_arc+1]) ? 1 : 0
    ns.succ_nums[ns.u_in+1] = oldsucc_nums
    
    # Set limits for updating last_succs
    up_limit_in = N(-1)
    up_limit_out = N(-1)
    if ns.last_succs[ns.join+1] == ns.v_in  # Compare 0-based
        up_limit_out = ns.join  # 0-based
    else
        up_limit_in = ns.join  # 0-based
    end
    
    # Update last_succs from v_in towards root
    u = ns.v_in  # 0-based
    @inbounds while u != up_limit_in && ns.last_succs[u+1] == ns.v_in
        ns.last_succs[u+1] = ns.last_succs[ns.u_out+1]  # Store 0-based
        u = ns.parents[u+1]  # Get 0-based parent
        if u == -1  # Check for no parent
            break
        end
    end
    
    # Update last_succs from v_out towards root
    @inbounds if ns.join != oldrev_threads && ns.v_in != oldrev_threads
        u = ns.v_out  # 0-based
        while u != up_limit_out && u != -1 && ns.last_succs[u+1] == oldlast_succs
            ns.last_succs[u+1] = oldrev_threads  # Store 0-based
            u = ns.parents[u+1]  # Get 0-based parent
        end
    else
        u = ns.v_out  # 0-based
        while u != up_limit_out && u != -1 && ns.last_succs[u+1] == oldlast_succs
            ns.last_succs[u+1] = ns.last_succs[ns.u_out+1]  # Store 0-based
            u = ns.parents[u+1]  # Get 0-based parent
        end
    end
    
    # Update succ_nums from v_in to join
    u = ns.v_in  # 0-based
    @inbounds while u != ns.join && u != -1
        ns.succ_nums[u+1] += oldsucc_nums
        u = ns.parents[u+1]  # Get 0-based parent
    end

    # Update succ_nums from v_out to join
    u = ns.v_out  # 0-based
    @inbounds while u != ns.join && u != -1
        ns.succ_nums[u+1] -= oldsucc_nums
        u = ns.parents[u+1]  # Get 0-based parent
    end
end

@inline function updatePotential!(ns::NetworkSimplex{V}) where V
    @inbounds @fastmath begin
        sigma = if ns.forwards[ns.u_in+1] != 0
            ns.pis[ns.v_in+1] - ns.pis[ns.u_in+1] - ns.costs[ns.preds[ns.u_in+1]+1]
        else
            ns.pis[ns.v_in+1] - ns.pis[ns.u_in+1] + ns.costs[ns.preds[ns.u_in+1]+1]
        end

        # Update potentials in moved subtree
        end_node = ns.threads[ns.last_succs[ns.u_in+1]+1]  # 0-based
        u = ns.u_in  # 0-based
        while u != end_node
            ns.pis[u+1] += sigma
            u = ns.threads[u+1]  # Get next 0-based node
        end
    end
end

# Arc reordering function for better cache locality
@inline function reorderArcsBySource!(ns::NetworkSimplex{V,A,N}) where {V,A,N}
    arc_num = arcNum(ns)

    # Create index array sorted by source node
    indices = collect(1:arc_num)
    sort!(indices, by = i -> ns.sources[i])

    # Create temporary arrays for reordered data
    new_sources = similar(ns.sources, arc_num)
    new_targets = similar(ns.targets, arc_num)
    new_costs = similar(ns.costs, arc_num)

    # Reorder arrays according to sorted indices
    @inbounds for (new_idx, old_idx) in enumerate(indices)
        new_sources[new_idx] = ns.sources[old_idx]
        new_targets[new_idx] = ns.targets[old_idx]
        new_costs[new_idx] = ns.costs[old_idx]
    end

    # Copy back to original arrays
    @inbounds @simd for i in 1:arc_num
        ns.sources[i] = new_sources[i]
        ns.targets[i] = new_targets[i]
        ns.costs[i] = new_costs[i]
    end

    return nothing
end

# Accessor functions
weights(ns::NetworkSimplex) = ns.supplies
dists(ns::NetworkSimplex) = ns.costs
total_cost(ns::NetworkSimplex) = ns.total_cost
n_iter(ns::NetworkSimplex) = ns.n_iter
flows(ns::NetworkSimplex) = ns.flows
potentials(ns::NetworkSimplex) = ns.pis



# ========== Pairwise Distance Computation ==========



using LinearAlgebra
using LoopVectorization
using StaticArrays

export AbstractPairwiseDistance, DefaultPairwiseDistance, compute_distance_matrix!,
       compute_distance, compute_distance_2d, AngularDistance, compute_angular_distance_2d,
       compute_distance_static

# Abstract base type for pairwise distance calculations
abstract type AbstractPairwiseDistance{V<:AbstractFloat} end

# Default implementation using Euclidean distance
mutable struct DefaultPairwiseDistance{V<:AbstractFloat} <: AbstractPairwiseDistance{V}
    R::V
    R2::V
    beta::V
    halfbeta::V
    
    function DefaultPairwiseDistance{V}(R::V=one(V), beta::V=one(V)) where V
        if R <= 0
            throw(ArgumentError("R must be positive"))
        end
        if beta < 0
            throw(ArgumentError("beta must be non-negative"))
        end
        new{V}(R, R*R, beta, beta/2)
    end
end

DefaultPairwiseDistance(R::Real=1.0, beta::Real=1.0) = 
    DefaultPairwiseDistance{Float64}(Float64(R), Float64(beta))

# Compute plain squared distance between two particles (SIMD optimized with LoopVectorization)
function plain_distance(p0::AbstractVector{V}, p1::AbstractVector{V}) where V
    dist = zero(V)
    @turbo for i in eachindex(p0, p1)
        diff = p0[i] - p1[i]
        dist += diff * diff
    end
    return dist
end

# Specialized version for StaticArrays - extremely fast for small dimensions
@inline function plain_distance(p0::SVector{N,V}, p1::SVector{N,V}) where {N,V}
    diff = p0 - p1  # SIMD optimized by StaticArrays
    return dot(diff, diff)  # Uses optimized dot product
end

# Generated function for specialized distance computation based on beta value
@generated function compute_distance(pd::DefaultPairwiseDistance{V}, p0::AbstractVector, p1::AbstractVector, ::Val{beta}) where {V, beta}
    if beta == 1.0
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath sqrt(plain_dist) / pd.R
        end
    elseif beta == 2.0
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath plain_dist / pd.R2
        end
    else
        return quote
            plain_dist = plain_distance(p0, p1)
            @fastmath (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Fallback for runtime beta values
@inline function compute_distance(pd::DefaultPairwiseDistance{V}, p0::AbstractVector, p1::AbstractVector) where V
    plain_dist = plain_distance(p0, p1)
    return compute_distance_from_plain(pd, plain_dist)
end

# Helper function to compute distance from plain distance
@inline function compute_distance_from_plain(pd::DefaultPairwiseDistance{V}, plain_dist::V) where V
    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Generated function for specialized 2D distance computation
@generated function compute_distance_2d(pd::DefaultPairwiseDistance{V}, p0_1::V, p0_2::V, p1_1::V, p1_2::V, ::Val{beta}) where {V, beta}
    if beta == 1.0
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                sqrt(plain_dist) / pd.R
            end
        end
    elseif beta == 2.0
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                plain_dist / pd.R2
            end
        end
    else
        return quote
            @fastmath begin
                diff1 = p0_1 - p1_1
                diff2 = p0_2 - p1_2
                plain_dist = muladd(diff1, diff1, diff2 * diff2)
                (plain_dist / pd.R2)^pd.halfbeta
            end
        end
    end
end

# Fallback for runtime beta values
@inline function compute_distance_2d(pd::DefaultPairwiseDistance{V}, p0_1::V, p0_2::V, p1_1::V, p1_2::V) where V
    @fastmath begin
        diff1 = p0_1 - p1_1
        diff2 = p0_2 - p1_2
        plain_dist = muladd(diff1, diff1, diff2 * diff2)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Fill distance matrix for two particle collections
function compute_distance_matrix!(pd::AbstractPairwiseDistance{V},
                                  particles0::Matrix{V},
                                  particles1::Matrix{V},
                                  dists::Vector{V},
                                  extra_particle::Symbol=:neither,
                                  extra_cost::V=one(V)) where V

    n0_orig = size(particles0, 2)
    n1_orig = size(particles1, 2)

    # Determine actual dimensions including dummy
    n0 = n0_orig + (extra_particle == :zero ? 1 : 0)
    n1 = n1_orig + (extra_particle == :one ? 1 : 0)
    
    # Compute pairwise distances
    k = 1

    if extra_particle == :zero
        # Extra particle added to event 0 (as last particle)
        # Fill matrix row by row: first n0_orig real particles, then dummy

        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            # First n0_orig rows: real particles from event0
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
            end
            # Last row: distances from dummy particle to all event1 particles
            @inbounds for j in 1:n1_orig
                dists[k] = extra_cost
                k += 1
            end
        else
            # First n0_orig rows: real particles from event0
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
            end
            # Last row: distances from dummy particle to all event1 particles
            @inbounds for j in 1:n1_orig
                dists[k] = extra_cost
                k += 1
            end
        end
    elseif extra_particle == :one
        # Extra particle added to event 1 (as last particle)
        # Fill matrix row by row: for each event0 particle, all n1_orig real + 1 dummy

        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            @inbounds for i in 1:n0_orig
                # First n1_orig columns: real particles from event1
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
                # Last column: distance to dummy particle
                dists[k] = extra_cost
                k += 1
            end
        else
            @inbounds for i in 1:n0_orig
                # First n1_orig columns: real particles from event1
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
                # Last column: distance to dummy particle
                dists[k] = extra_cost
                k += 1
            end
        end
    else
        # No extra particle - standard case
        if size(particles0, 1) == 2 && isa(pd, DefaultPairwiseDistance{V})  # Optimize for 2D case
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dists[k] = compute_distance_2d(pd, particles0[1,i], particles0[2,i],
                                                       particles1[1,j], particles1[2,j])
                    k += 1
                end
            end
        else
            @inbounds for i in 1:n0_orig
                for j in 1:n1_orig
                    dist_val = zero(V)
                    @fastmath @inbounds @simd for d in 1:size(particles0, 1)
                        diff = particles0[d, i] - particles1[d, j]
                        dist_val = muladd(diff, diff, dist_val)
                    end
                    @fastmath dists[k] = compute_distance_from_plain(pd, dist_val)
                    k += 1
                end
            end
        end
    end

    return nothing
end

# Specialized distances for physics applications

# Angular distance for particles with (eta, phi) coordinates
struct AngularDistance{V<:AbstractFloat} <: AbstractPairwiseDistance{V}
    R::V
    R2::V
    beta::V
    halfbeta::V
    
    function AngularDistance{V}(R::V=one(V), beta::V=one(V)) where V
        if R <= 0
            throw(ArgumentError("R must be positive"))
        end
        if beta < 0
            throw(ArgumentError("beta must be non-negative"))
        end
        new{V}(R, R*R, beta, beta/2)
    end
end

@inline function compute_distance(pd::AngularDistance{V}, p0::AbstractVector, p1::AbstractVector) where V
    @fastmath @inbounds begin
        # Assuming p0 and p1 have (eta, phi) coordinates
        deta = p0[1] - p1[1]
        dphi = p0[2] - p1[2]

        # Optimized phi wrapping using modulo
        dphi = mod(dphi + π, 2π) - π

        # Compute angular distance
        plain_dist = muladd(deta, deta, dphi * dphi)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# Specialized fast version for angular distance with direct coordinate access
@inline function compute_angular_distance_2d(pd::AngularDistance{V}, eta0::V, phi0::V, eta1::V, phi1::V) where V
    @fastmath begin
        deta = eta0 - eta1
        dphi = mod(phi0 - phi1 + π, 2π) - π
        plain_dist = muladd(deta, deta, dphi * dphi)  # Fused multiply-add

        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

# High-performance static distance computation for known dimensions
@inline function compute_distance_static(pd::DefaultPairwiseDistance{V}, p0::SVector{2,V}, p1::SVector{2,V}) where V
    diff = p0 - p1
    plain_dist = dot(diff, diff)

    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end

@inline function compute_distance_static(pd::DefaultPairwiseDistance{V}, p0::SVector{3,V}, p1::SVector{3,V}) where V
    diff = p0 - p1
    plain_dist = dot(diff, diff)

    @fastmath begin
        if pd.beta == 1.0
            return sqrt(plain_dist) / pd.R
        elseif pd.beta == 2.0
            return plain_dist / pd.R2
        else
            return (plain_dist / pd.R2)^pd.halfbeta
        end
    end
end



# ========== Event Handling ==========



export AbstractEvent, Event, EuclideanParticleEvent, total_weight, normalize_weights!, ensure_weights

# Abstract base type for events
abstract type AbstractEvent{V<:AbstractFloat} end

# Basic Event structure
mutable struct Event{V<:AbstractFloat} <: AbstractEvent{V}
    particles::Matrix{V}  # Each column is a particle
    weights::Vector{V}
    event_weight::V
    total_weight::V
    has_weights::Bool
    
    # Constructor from particles only
    function Event(particles::Matrix{V}, event_weight::V=one(V)) where V
        n_particles = size(particles, 2)
        new{V}(particles, Vector{V}(), event_weight, zero(V), false)
    end
    
    # Constructor from weights and particles
    function Event(weights::Vector{V}, particles::Matrix{V}, event_weight::V=one(V)) where V
        if length(weights) != size(particles, 2)
            throw(DimensionMismatch("Number of weights must match number of particles"))
        end
        total = sum(weights)
        new{V}(particles, weights, event_weight, total, true)
    end
end

# Euclidean Particle Event (particles with weights embedded)
mutable struct EuclideanParticleEvent{V<:AbstractFloat} <: AbstractEvent{V}
    particles::Matrix{V}
    weights::Vector{V}
    event_weight::V
    total_weight::V
    has_weights::Bool
    
    function EuclideanParticleEvent(particles::Matrix{V}, event_weight::V=one(V)) where V
        n_particles = size(particles, 2)
        # Extract weights from first row of particles (common in physics applications)
        weights = particles[1, :]
        total = sum(weights)
        new{V}(particles, weights, event_weight, total, true)
    end
end

# Helper functions
function total_weight(ev::AbstractEvent)
    if !ev.has_weights
        throw(ErrorException("Event does not have weights"))
    end
    return ev.total_weight
end

function ensure_weights(ev::AbstractEvent{V}) where V
    if !ev.has_weights
        # Default: equal weights
        n_particles = size(ev.particles, 2)
        ev.weights = ones(V, n_particles) / n_particles
        ev.total_weight = one(V)
        ev.has_weights = true
    end
end

function normalize_weights!(ev::AbstractEvent{V}) where V
    if !ev.has_weights
        throw(ErrorException("Weights must be set prior to normalization"))
    end
    
    if ev.total_weight ≈ 0
        throw(ErrorException("Cannot normalize zero total weight"))
    end
    
    # Normalize each weight
    norm_total = zero(V)
    for i in eachindex(ev.weights)
        ev.weights[i] /= ev.total_weight
        norm_total += ev.weights[i]
    end
    ev.total_weight = norm_total
end

# Dimension of particles
dimension(ev::AbstractEvent) = size(ev.particles, 1)

# Number of particles
n_particles(ev::AbstractEvent) = size(ev.particles, 2)



# ========== EMD Computation ==========



using LinearAlgebra

# Get modules from parent





export EMD, compute_emd, ExtraParticle, EMDStatus, flow_matrix, Success
export Event, EuclideanParticleEvent, total_weight

@enum ExtraParticle::Int8 begin
    Neither = -1
    Zero = 0
    One = 1
end

# Main EMD class
mutable struct EMD{V<:AbstractFloat}
    # Parameters
    R::V
    beta::V
    norm::Bool
    external_dists::Bool
    
    # Components
    pairwise_distance::DefaultPairwiseDistance{V}
    network_simplex::NetworkSimplex{V,Int32,Int32,UInt8}
    
    # State variables
    n0::Int
    n1::Int
    extra::ExtraParticle
    weightdiff::V
    scale::V
    emd::V
    status::EMDStatus
    
    function EMD{V}(;R::V=one(V), beta::V=one(V), norm::Bool=false,
                    external_dists::Bool=false,
                    n_iter_max::Int=100000,
                    epsilon_large_factor::V=V(1000),
                    epsilon_small_factor::V=one(V)) where V
        
        pd = DefaultPairwiseDistance{V}(R, beta)
        ns = NetworkSimplex{V,Int32,Int32,UInt8}(n_iter_max, epsilon_large_factor, epsilon_small_factor)
        
        new{V}(R, beta, norm, external_dists, pd, ns,
               0, 0, Neither, zero(V), one(V), zero(V), Success)
    end
end

EMD(;kwargs...) = EMD{Float64}(;kwargs...)

function compute_emd(emd::EMD{V}, ev0::AbstractEvent{V}, ev1::AbstractEvent{V}) where V
    # Ensure weights exist
    ensure_weights(ev0)
    ensure_weights(ev1)
    
    # Normalize weights if requested (before calculating weight difference)
    if emd.norm
        normalize_weights!(ev0)
        normalize_weights!(ev1)
    end
    
    ws0 = ev0.weights
    ws1 = ev1.weights
    
    # Get number of particles
    emd.n0 = length(ws0)
    emd.n1 = length(ws1)
    
    # Handle weight difference
    emd.weightdiff = total_weight(ev1) - total_weight(ev0)
    
    # Setup weights vector in network simplex
    weights_vec = weights(emd.network_simplex)
    needed_size = emd.n0 + emd.n1 + 1 + (emd.weightdiff != 0 ? 1 : 0)

    # Only resize if needed
    if length(weights_vec) < needed_size
        resize!(weights_vec, needed_size)
    end

    if emd.norm || emd.external_dists || emd.weightdiff ≈ 0
        # No extra particle needed
        emd.extra = Neither
        @inbounds copyto!(weights_vec, 1, ws0, 1, emd.n0)
        @inbounds copyto!(weights_vec, emd.n0 + 1, ws1, 1, emd.n1)
    elseif emd.weightdiff > 0
        # Add extra particle to event0
        emd.extra = Zero
        emd.n0 += 1
        @inbounds copyto!(weights_vec, 1, ws0, 1, length(ws0))
        @inbounds weights_vec[length(ws0) + 1] = emd.weightdiff
        @inbounds copyto!(weights_vec, length(ws0) + 2, ws1, 1, length(ws1))
    else
        # Add extra particle to event1
        emd.extra = One
        emd.n1 += 1
        @inbounds copyto!(weights_vec, 1, ws0, 1, length(ws0))
        @inbounds copyto!(weights_vec, length(ws0) + 1, ws1, 1, length(ws1))
        @inbounds weights_vec[length(ws0) + length(ws1) + 1] = -emd.weightdiff
    end
    
    # Set scale and scale weights if not normalized
    if emd.norm
        emd.scale = one(V)  # Already normalized above
    else
        # Scale weights if not normalized
        emd.scale = max(total_weight(ev0), total_weight(ev1))
        weights_vec ./= emd.scale
    end
    
    # Compute distances if not external
    if !emd.external_dists
        dists_vec = dists(emd.network_simplex)
        needed_dist_size = emd.n0 * emd.n1
        if length(dists_vec) < needed_dist_size
            resize!(dists_vec, needed_dist_size)
        end

        # Use original particle arrays without copying (no hcat needed!)
        particles0 = ev0.particles
        particles1 = ev1.particles

        # The compute_distance_matrix! function will handle the dummy particle
        # by writing the extra cost directly into dists_vec
        extra_sym = emd.extra == Zero ? :zero : (emd.extra == One ? :one : :neither)
        # Use 1.0 as extra_cost to maintain consistency with previous implementation
        compute_distance_matrix!(emd.pairwise_distance, particles0, particles1, dists_vec, extra_sym, one(V))
    end
    
    # Run network simplex
    emd.status = compute!(emd.network_simplex, emd.n0, emd.n1)
    emd.emd = total_cost(emd.network_simplex)
    
    # Scale result if not normalized
    if emd.status == Success && !emd.norm
        emd.emd *= emd.scale
    end
    
    return emd.emd, emd.status
end

# Simple interface function
function compute_emd(weights0::Vector{V}, weights1::Vector{V}, 
                    particles0::Matrix{V}, particles1::Matrix{V};
                    R::V=one(V), beta::V=one(V), norm::Bool=false,
                    n_iter_max::Int=100000) where V<:AbstractFloat
    
    ev0 = Event(weights0, particles0)
    ev1 = Event(weights1, particles1)
    
    emd_obj = EMD{V}(R=R, beta=beta, norm=norm, n_iter_max=n_iter_max)
    
    return compute_emd(emd_obj, ev0, ev1)
end

# Access flow matrix
function flow_matrix(emd::EMD{V}) where V
    flow_vec = flows(emd.network_simplex)
    flow_mat = zeros(V, emd.n0, emd.n1)
    
    idx = 1
    for i in 1:emd.n0
        for j in 1:emd.n1
            flow_mat[i, j] = flow_vec[idx] * emd.scale
            idx += 1
        end
    end
    
    return flow_mat
end

# Get/set parameters
set_R(emd::EMD{V}, R::V) where V = (emd.R = R; emd.pairwise_distance.R = R; emd.pairwise_distance.R2 = R*R)
set_beta(emd::EMD{V}, beta::V) where V = (emd.beta = beta; emd.pairwise_distance.beta = beta; emd.pairwise_distance.halfbeta = beta/2)

get_R(emd::EMD) = emd.R
get_beta(emd::EMD) = emd.beta



# ========== Pairwise EMD Computation ==========



using Base.Threads



export PairwiseEMD, compute_pairwise!, compute_pairwise_emd

mutable struct PairwiseEMD{V<:AbstractFloat}
    # EMD parameters
    R::V
    beta::V
    norm::Bool

    # Threading parameters
    num_threads::Int

    # EMD objects (one per thread)
    emd_objs::Vector{EMD{V}}

    # Storage for events
    events::Vector{Event{V}}

    # Distance matrix storage
    distances::Matrix{V}

    function PairwiseEMD{V}(;R::V=one(V), beta::V=one(V), norm::Bool=false,
                            num_threads::Int=Threads.nthreads(),
                            n_iter_max::Int=100000,
                            epsilon_large_factor::V=V(1000),
                            epsilon_small_factor::V=one(V)) where V

        # Create EMD objects for each thread
        emd_objs = [EMD{V}(R=R, beta=beta, norm=norm, n_iter_max=n_iter_max,
                           epsilon_large_factor=epsilon_large_factor,
                           epsilon_small_factor=epsilon_small_factor)
                    for _ in 1:num_threads]

        new{V}(R, beta, norm, num_threads, emd_objs, Event{V}[], Matrix{V}(undef, 0, 0))
    end
end

PairwiseEMD(;kwargs...) = PairwiseEMD{Float64}(;kwargs...)

# Compute pairwise EMDs for a single set of events
function compute_pairwise!(pemd::PairwiseEMD{V}, events::Vector{Event{V}}) where V
    n = length(events)
    pemd.events = events
    pemd.distances = zeros(V, n, n)
    
    # Compute upper triangle (including diagonal)
    @threads for idx in 1:((n * (n + 1)) ÷ 2)
        # Convert linear index to (i, j) coordinates for upper triangle
        i = ceil(Int, (sqrt(1 + 8*idx) - 1) / 2)
        j = idx - (i * (i - 1)) ÷ 2
        
        if i == j
            pemd.distances[i, j] = zero(V)
        else
            thread_id = Threads.threadid()
            emd_val, status = compute_emd(pemd.emd_objs[thread_id], events[i], events[j])
            
            if status == Success
                pemd.distances[i, j] = emd_val
                pemd.distances[j, i] = emd_val  # Symmetric
            else
                pemd.distances[i, j] = V(NaN)
                pemd.distances[j, i] = V(NaN)
            end
        end
    end
    
    return pemd.distances
end

# Compute pairwise EMDs between two sets of events
function compute_pairwise!(pemd::PairwiseEMD{V}, eventsA::Vector{Event{V}}, eventsB::Vector{Event{V}}) where V
    nA = length(eventsA)
    nB = length(eventsB)
    
    pemd.events = vcat(eventsA, eventsB)
    pemd.distances = zeros(V, nA, nB)
    
    # Compute all pairs
    @threads for idx in 1:(nA * nB)
        i = ((idx - 1) ÷ nB) + 1
        j = ((idx - 1) % nB) + 1
        
        thread_id = Threads.threadid()
        emd_val, status = compute_emd(pemd.emd_objs[thread_id], eventsA[i], eventsB[j])
        
        if status == Success
            pemd.distances[i, j] = emd_val
        else
            pemd.distances[i, j] = V(NaN)
        end
    end
    
    return pemd.distances
end

# Simple interface for computing pairwise EMDs
function compute_pairwise_emd(weights_list::Vector{Vector{V}}, 
                              particles_list::Vector{Matrix{V}};
                              R::V=one(V), beta::V=one(V), norm::Bool=false,
                              num_threads::Int=Threads.nthreads()) where V<:AbstractFloat
    
    # Create events
    events = [Event(weights_list[i], particles_list[i]) for i in 1:length(weights_list)]
    
    # Create pairwise EMD object
    pemd = PairwiseEMD{V}(R=R, beta=beta, norm=norm, num_threads=num_threads)
    
    # Compute and return distances
    return compute_pairwise!(pemd, events)
end



# ========== Main API Functions ==========

"""
    WassersteinAPI

Julia API for Wasserstein/EMD distance computation, matching the Python energyflow interface.
"""


using LinearAlgebra
using Base.Threads

# Import from sibling modules








"""
    emd(ev0, ev1; dists=nothing, R=1.0, beta=1.0, norm=false, gdim=2, 
        mask=false, return_flow=false, do_timing=false,
        n_iter_max=100000, epsilon_large_factor=10000.0, epsilon_small_factor=1.0)

Compute the EMD between two events using the Wasserstein distance.

# Arguments
- `ev0::Matrix`: First event as (M, 1+gdim) array where M is multiplicity.
  First column is particle weights (e.g., pT), remaining columns are coordinates.
  For collider physics: (pT, y, φ) where y is rapidity, φ is azimuthal angle.
- `ev1::Matrix`: Second event in same format as ev0
- `dists::Union{Matrix,Nothing}=nothing`: Pre-computed distance matrix between particles.
  If nothing, Euclidean distance is computed from particle coordinates.
- `R::Float64=1.0`: EMD parameter controlling relative importance of transport vs creation/destruction.
  Must be ≥ half the maximum ground distance for valid metric.
- `beta::Float64=1.0`: Angular weighting exponent. Distance matrix is raised to this power.
- `norm::Bool=false`: Whether to normalize particle weights to sum to 1
- `gdim::Int=2`: Dimension of ground metric space (number of coordinate columns to use)
- `mask::Bool=false`: If true, mask particles farther than R from origin
- `return_flow::Bool=false`: Whether to return the optimal transport flow matrix
- `do_timing::Bool=false`: Whether to return computation time
- `n_iter_max::Int=100000`: Maximum iterations for network simplex solver
- `epsilon_large_factor::Float64=10000.0`: Tolerance factor for solver (multiplied by machine epsilon)
- `epsilon_small_factor::Float64=1.0`: Stricter tolerance factor for solver

# Returns
- `Float64`: The EMD value
- `Matrix{Float64}` (optional): Flow matrix if return_flow=true
- `Float64` (optional): Computation time in seconds if do_timing=true

# Examples
```julia
# Basic usage with 2D particles
ev0 = [0.5 1.0 0.0;   # (pT, y, φ)
       0.3 0.0 1.0;
       0.2 -1.0 -1.0]
ev1 = [0.4 1.1 0.1;
       0.35 0.1 0.9;
       0.25 -0.9 -1.1]

emd_val = emd(ev0, ev1)

# With flow matrix
emd_val, flow = emd(ev0, ev1, return_flow=true)

# With pre-computed distances
dists = compute_distance_matrix(ev0, ev1)
emd_val = emd(ev0, ev1, dists=dists)
```
"""
function emd_network_simplex(ev0::AbstractMatrix, ev1::AbstractMatrix;
            dists::Union{AbstractMatrix,Nothing}=nothing,
            R::Real=1.0, beta::Real=1.0, norm::Bool=false,
            gdim::Int=2, mask::Bool=false,
            return_flow::Bool=false, do_timing::Bool=false,
            n_iter_max::Int=100000,
            epsilon_large_factor::Real=10000.0,
            epsilon_small_factor::Real=1.0)
    
    # Convert to Float64 for computation
    R64 = Float64(R)
    beta64 = Float64(beta)
    epsilon_large64 = Float64(epsilon_large_factor)
    epsilon_small64 = Float64(epsilon_small_factor)
    
    # Extract weights and particles
    weights0 = Float64.(ev0[:, 1])
    weights1 = Float64.(ev1[:, 1])
    
    # Handle coordinates based on gdim
    if dists === nothing
        # Use only first gdim coordinate columns (plus weight column)
        coord_cols = min(gdim, size(ev0, 2) - 1)
        particles0 = Float64.(ev0[:, 2:(1+coord_cols)]')  # Transpose for column-per-particle
        particles1 = Float64.(ev1[:, 2:(1+coord_cols)]')
        
        # Apply mask if requested
        if mask
            # Mask particles farther than R from origin
            dist0 = [LinearAlgebra.norm(particles0[:, i]) for i in 1:size(particles0, 2)]
            dist1 = [LinearAlgebra.norm(particles1[:, i]) for i in 1:size(particles1, 2)]
            
            keep_mask0 = dist0 .<= R64
            keep_mask1 = dist1 .<= R64
            
            weights0 = weights0[keep_mask0]
            weights1 = weights1[keep_mask1]
            particles0 = particles0[:, keep_mask0]
            particles1 = particles1[:, keep_mask1]
        end
    else
        # Using pre-computed distances, create dummy particles
        particles0 = zeros(Float64, 1, length(weights0))
        particles1 = zeros(Float64, 1, length(weights1))
    end
    
    # Create EMD object with parameters
    emd_obj = EMD{Float64}(
        R=R64, beta=beta64, norm=norm,
        external_dists=(dists !== nothing),
        n_iter_max=n_iter_max,
        epsilon_large_factor=epsilon_large64,
        epsilon_small_factor=epsilon_small64
    )
    
    # Create events
    event0 = Event(weights0, particles0)
    event1 = Event(weights1, particles1)
    
    # Set external distances if provided
    if dists !== nothing
        # Copy distances to EMD object's internal storage
        dists_vec = NetworkSimplexModule.dists(emd_obj.network_simplex)
        resize!(dists_vec, length(weights0) * length(weights1))
        
        # Flatten the distance matrix into the vector
        idx = 1
        for i in 1:length(weights0)
            for j in 1:length(weights1)
                dists_vec[idx] = Float64(dists[i, j])
                idx += 1
            end
        end
    end
    
    # Start timing if requested
    start_time = do_timing ? time() : 0.0
    
    # Compute EMD
    emd_val, status = compute_emd(emd_obj, event0, event1)
    
    # Check status
    if status != Success
        throw(ErrorException("EMD computation failed with status: $status"))
    end
    
    # End timing
    computation_time = do_timing ? time() - start_time : 0.0
    
    # Prepare return values
    results = Any[emd_val]
    
    if return_flow
        flow = flow_matrix(emd_obj)
        push!(results, flow)
    end
    
    if do_timing
        push!(results, computation_time)
    end
    
    return length(results) == 1 ? results[1] : tuple(results...)
end

"""
    emds(events0, events1=nothing; R=1.0, beta=1.0, norm=false, gdim=2,
         mask=false, n_jobs=-1, print_every=0, verbose=0,
         throw_on_error=true, n_iter_max=100000,
         epsilon_large_factor=10000.0, epsilon_small_factor=1.0)

Compute EMDs between collections of events.

# Arguments
- `events0::Vector{Matrix}`: Collection of events, each as (M, 1+gdim) array
- `events1::Union{Vector{Matrix},Nothing}=nothing`: Second collection of events.
  If nothing, computes pairwise distances within events0 (symmetric matrix)
- `R::Float64=1.0`: EMD parameter
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `gdim::Int=2`: Dimension of ground metric space
- `mask::Bool=false`: Mask particles farther than R from origin
- `n_jobs::Int=-1`: Number of threads (-1 uses all available)
- `print_every::Int=0`: Print progress every N computations (0 = no printing)
- `verbose::Int=0`: Verbosity level
- `throw_on_error::Bool=true`: Whether to throw exceptions on errors
- `n_iter_max::Int=100000`: Maximum iterations for solver
- `epsilon_large_factor::Float64=10000.0`: Tolerance factor
- `epsilon_small_factor::Float64=1.0`: Stricter tolerance factor

# Returns
- `Matrix{Float64}`: EMD values. Shape (len(events0), len(events0)) if events1=nothing (symmetric),
  otherwise (len(events0), len(events1))

# Examples
```julia
# Compute all pairwise EMDs within a set
events = [rand(10, 3) for _ in 1:5]  # 5 events with 10 particles each
dist_matrix = emds(events)  # 5x5 symmetric matrix

# Compute EMDs between two different sets
events_a = [rand(10, 3) for _ in 1:4]
events_b = [rand(10, 3) for _ in 1:6]
dist_matrix = emds(events_a, events_b)  # 4x6 matrix
```
"""
function emds_network_simplex(events0::Vector{<:AbstractMatrix}, 
             events1::Union{Vector{<:AbstractMatrix},Nothing}=nothing;
             R::Real=1.0, beta::Real=1.0, norm::Bool=false,
             gdim::Int=2, mask::Bool=false,
             n_jobs::Int=-1, print_every::Int=0, verbose::Int=0,
             throw_on_error::Bool=true, n_iter_max::Int=100000,
             epsilon_large_factor::Real=10000.0,
             epsilon_small_factor::Real=1.0)
    
    # Convert parameters to Float64
    R64 = Float64(R)
    beta64 = Float64(beta)
    epsilon_large64 = Float64(epsilon_large_factor)
    epsilon_small64 = Float64(epsilon_small_factor)
    
    # Determine number of threads
    num_threads = n_jobs == -1 ? Threads.nthreads() : n_jobs
    
    # Process events into weights and particles
    function process_events(events)
        weights_list = Vector{Vector{Float64}}()
        particles_list = Vector{Matrix{Float64}}()
        
        for ev in events
            weights = Float64.(ev[:, 1])
            
            # Use only first gdim coordinate columns
            coord_cols = min(gdim, size(ev, 2) - 1)
            particles = Float64.(ev[:, 2:(1+coord_cols)]')  # Transpose
            
            # Apply mask if requested
            if mask
                dists = [LinearAlgebra.norm(particles[:, i]) for i in 1:size(particles, 2)]
                keep_idx = dists .<= R64
                weights = weights[keep_idx]
                particles = particles[:, keep_idx]
            end
            
            push!(weights_list, weights)
            push!(particles_list, particles)
        end
        
        return weights_list, particles_list
    end
    
    weights0_list, particles0_list = process_events(events0)
    
    if events1 === nothing
        # Symmetric case - compute within events0
        if verbose > 0
            println("Computing $(length(events0))×$(length(events0)) symmetric EMD matrix...")
        end
        
        # Create PairwiseEMD object with correct number of threads
        actual_threads = min(num_threads, Threads.nthreads())
        pemd = PairwiseEMD{Float64}(
            R=R64, beta=beta64, norm=norm,
            num_threads=actual_threads,
            n_iter_max=n_iter_max,
            epsilon_large_factor=epsilon_large64,
            epsilon_small_factor=epsilon_small64
        )
        
        # Create Event objects
        event_objs = [Event(weights0_list[i], particles0_list[i]) 
                      for i in 1:length(weights0_list)]
        
        # Compute pairwise distances
        n = length(event_objs)
        dist_matrix = zeros(Float64, n, n)
        
        # Progress tracking
        total_pairs = (n * (n - 1)) ÷ 2
        completed = Threads.Atomic{Int}(0)
        last_print = Threads.Atomic{Int}(0)
        
        # Parallel computation
        pair_idx = 0
        pairs = [(i, j) for i in 1:n for j in (i+1):n]

        @threads for (i, j) in pairs
                thread_id = min(Threads.threadid(), length(pemd.emd_objs))
                emd_val, status = compute_emd(pemd.emd_objs[thread_id],
                                             event_objs[i], event_objs[j])
                
                if status != Success
                    if throw_on_error
                        throw(ErrorException("EMD computation failed for pair ($i,$j) with status: $status"))
                    else
                        emd_val = NaN
                    end
                end
                
                dist_matrix[i, j] = emd_val
                dist_matrix[j, i] = emd_val  # Symmetric

            # Progress reporting
            Threads.atomic_add!(completed, 1)
            if verbose > 0 && print_every > 0
                comp_val = completed[]
                if comp_val % print_every == 0 || comp_val == total_pairs
                    if comp_val > last_print[]
                        println("  Completed $comp_val/$total_pairs pairs")
                        last_print[] = comp_val
                    end
                end
            end
        end
        
        return dist_matrix
        
    else
        # Asymmetric case - compute between events0 and events1
        weights1_list, particles1_list = process_events(events1)
        
        if verbose > 0
            println("Computing $(length(events0))×$(length(events1)) EMD matrix...")
        end
        
        # Create PairwiseEMD object with correct number of threads
        actual_threads = min(num_threads, Threads.nthreads())
        pemd = PairwiseEMD{Float64}(
            R=R64, beta=beta64, norm=norm,
            num_threads=actual_threads,
            n_iter_max=n_iter_max,
            epsilon_large_factor=epsilon_large64,
            epsilon_small_factor=epsilon_small64
        )
        
        # Create Event objects
        events0_objs = [Event(weights0_list[i], particles0_list[i]) 
                        for i in 1:length(weights0_list)]
        events1_objs = [Event(weights1_list[i], particles1_list[i]) 
                        for i in 1:length(weights1_list)]
        
        # Compute all pairs
        nA = length(events0_objs)
        nB = length(events1_objs)
        dist_matrix = zeros(Float64, nA, nB)
        
        # Progress tracking
        total_pairs = nA * nB
        completed = Threads.Atomic{Int}(0)
        last_print = Threads.Atomic{Int}(0)
        
        # Parallel computation - ensure thread_id doesn't exceed available EMD objects
        @threads for idx in 1:total_pairs
            i = ((idx - 1) ÷ nB) + 1
            j = ((idx - 1) % nB) + 1

            thread_id = min(Threads.threadid(), length(pemd.emd_objs))
            emd_val, status = compute_emd(pemd.emd_objs[thread_id],
                                         events0_objs[i], events1_objs[j])
            
            if status != Wasserstein.Success
                if throw_on_error
                    throw(ErrorException("EMD computation failed for pair ($i,$j) with status: $status"))
                else
                    emd_val = NaN
                end
            end
            
            dist_matrix[i, j] = emd_val
            
            # Progress reporting
            Threads.atomic_add!(completed, 1)
            if verbose > 0 && print_every > 0
                comp_val = completed[]
                if comp_val % print_every == 0 || comp_val == total_pairs
                    if comp_val > last_print[]
                        println("  Completed $comp_val/$total_pairs pairs")
                        last_print[] = comp_val
                    end
                end
            end
        end
        
        return dist_matrix
    end
end

# Convenience function for single event format conversion
"""
    format_event(weights, coordinates)

Convert separate weights and coordinates into event matrix format.

# Arguments
- `weights::Vector`: Particle weights (e.g., pT values)
- `coordinates::Matrix`: Particle coordinates, each column is a particle

# Returns
- `Matrix`: Event in (M, 1+gdim) format for EMD computation
"""
function format_event(weights::AbstractVector, coordinates::AbstractMatrix)
    M = length(weights)
    gdim = size(coordinates, 1)
    
    if size(coordinates, 2) != M
        throw(DimensionMismatch("Number of weights must match number of particles"))
    end
    
    event = zeros(M, 1 + gdim)
    event[:, 1] = weights
    event[:, 2:end] = coordinates'
    
    return event
end

export format_event



# ========== EnergyFlow.jl Export Functions ==========

"""
    emd_network_simplex(event1::Matrix{Float64}, event2::Matrix{Float64}; kwargs...)

Compute EMD between two events using the fast network simplex solver.

# Arguments
- `event1`: First event as (M, 3) or (M, 4) matrix [pT, y, phi] or [pT, y, phi, weight]
- `event2`: Second event as (N, 3) or (N, 4) matrix

# Keywords
- `R::Float64=1.0`: R parameter for EMD
- `beta::Float64=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `return_flow::Bool=false`: Whether to return flow matrix
- `n_iter_max::Int=100000`: Maximum iterations for solver

# Returns
- `Float64`: EMD value
- `Matrix{Float64}` (optional): Flow matrix if return_flow=true
"""
# The emd_network_simplex function continues below...

# The emds_network_simplex function continues below...

