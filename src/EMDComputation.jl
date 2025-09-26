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
                              num_threads::Int=Threads.nthreads(),
                              n_iter_max::Int=100000) where V<:AbstractFloat

    # Create events
    events = [Event(weights_list[i], particles_list[i]) for i in 1:length(weights_list)]

    # Create PairwiseEMD object
    pemd = PairwiseEMD{V}(R=R, beta=beta, norm=norm,
                          num_threads=num_threads,
                          n_iter_max=n_iter_max)

    # Compute and return
    return compute_pairwise!(pemd, events)
end
