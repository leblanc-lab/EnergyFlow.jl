


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

