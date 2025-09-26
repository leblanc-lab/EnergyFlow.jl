# NetworkSimplexSolver.jl - Main module for network simplex-based EMD computation

module Wasserstein

using LinearAlgebra
using LoopVectorization
using StaticArrays
using Base.Threads

# Include component modules
include("NetworkSimplexCore.jl")
include("DistanceMetrics.jl")
include("EventTypes.jl")
include("EMDComputation.jl")

# Export main user-facing functions
export emd_network_simplex, emds_network_simplex

# Re-export from submodules
# From NetworkSimplexCore
export NetworkSimplex, compute!, EMDStatus, Success, Empty, SupplyMismatch, Unbounded, MaxIterReached, Infeasible, NSf32, NSf64

# From DistanceMetrics
export AbstractPairwiseDistance, DefaultPairwiseDistance, compute_distance_matrix!

# From EventTypes
export AbstractEvent, Event, EuclideanParticleEvent, total_weight, normalize_weights!, ensure_weights

# From EMDComputation
export EMD, compute_emd, ExtraParticle, flow_matrix
export PairwiseEMD, compute_pairwise!, compute_pairwise_emd

# Utility function
export format_event

"""
    format_event(weights::AbstractVector, coordinates::AbstractMatrix)

Format weights and coordinates into an event matrix for EMD computation.
Returns a matrix where first column is weights, remaining columns are coordinates.
"""
function format_event(weights::AbstractVector, coordinates::AbstractMatrix)
    if length(weights) != size(coordinates, 2)
        throw(DimensionMismatch("Number of weights must match number of particles"))
    end

    # Create matrix with weights in first column
    result = zeros(eltype(weights), length(weights), 1 + size(coordinates, 1))
    result[:, 1] = weights
    result[:, 2:end] = coordinates'

    return result
end

# ========== Main EMD Interface Functions ==========

"""
    emd_network_simplex(ev0::AbstractMatrix, ev1::AbstractMatrix; kwargs...)

Compute Earth Mover's Distance between two events using network simplex algorithm.

# Arguments
- `ev0`, `ev1`: Events as matrices where first column is weights, rest are coordinates

# Keywords
- `dists::Union{AbstractMatrix,Nothing}=nothing`: Pre-computed distance matrix
- `R::Real=1.0`: Maximum distance parameter
- `beta::Real=1.0`: Angular weighting exponent
- `norm::Bool=false`: Whether to normalize weights
- `gdim::Int=2`: Dimension of ground space (2D or 3D)
- `mask::Bool=false`: Whether to apply distance masking with R
- `return_flow::Bool=false`: Whether to return flow matrix
- `do_timing::Bool=false`: Whether to time computation
- `n_iter_max::Int=100000`: Maximum iterations
- `epsilon_large_factor::Real=10000.0`: Large epsilon factor
- `epsilon_small_factor::Real=1.0`: Small epsilon factor

# Returns
- `emd_value::Float64`: The EMD value
- `status::EMDStatus`: Status of computation (optional, if return_flow=true also returns flow matrix)

# Examples
```julia
# Basic usage
ev0 = [1.0 0.0 0.0; 1.0 0.5 0.5]  # 2 particles: [weight y phi]
ev1 = [1.0 0.1 0.1; 1.0 0.6 0.6]
emd_val = emd_network_simplex(ev0, ev1)

# With pre-computed distances
dists = compute_distance_matrix(ev0, ev1)
emd_val = emd_network_simplex(ev0, ev1, dists=dists)
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
    else
        # When using pre-computed distances, we don't need particle coordinates
        particles0 = Matrix{Float64}(undef, 0, 0)
        particles1 = Matrix{Float64}(undef, 0, 0)
    end

    # Apply mask if requested
    if mask && dists === nothing
        # Only keep particles within distance R from origin
        dists0 = [LinearAlgebra.norm(particles0[:, i]) for i in 1:size(particles0, 2)]
        dists1 = [LinearAlgebra.norm(particles1[:, i]) for i in 1:size(particles1, 2)]

        keep0 = dists0 .<= R64
        keep1 = dists1 .<= R64

        weights0 = weights0[keep0]
        weights1 = weights1[keep1]
        particles0 = particles0[:, keep0]
        particles1 = particles1[:, keep1]
    end

    # Create Event objects
    ev0_obj = Event(weights0, particles0)
    ev1_obj = Event(weights1, particles1)

    # Create EMD object with parameters
    emd_obj = EMD{Float64}(
        R=R64, beta=beta64, norm=norm,
        external_dists=(dists !== nothing),
        n_iter_max=n_iter_max,
        epsilon_large_factor=epsilon_large64,
        epsilon_small_factor=epsilon_small64
    )

    # Set external distances if provided
    if dists !== nothing
        # Copy distances to network simplex dists vector
        dists_vec = dists(emd_obj.network_simplex)
        dists_flat = vec(Float64.(dists'))  # Transpose and flatten
        resize!(dists_vec, length(dists_flat))
        copyto!(dists_vec, dists_flat)
    end

    # Compute EMD
    if do_timing
        start_time = time()
    end

    emd_val, status = compute_emd(emd_obj, ev0_obj, ev1_obj)

    if do_timing
        elapsed = time() - start_time
        println("EMD computation time: $elapsed seconds")
    end

    if return_flow
        flow_mat = flow_matrix(emd_obj)
        return emd_val, status, flow_mat
    else
        return emd_val
    end
end

"""
    emds_network_simplex(events::Vector{<:AbstractMatrix}; kwargs...)

Compute symmetric pairwise EMD matrix for a collection of events.

# Arguments
- `events`: Vector of event matrices

# Keywords
Same as `emd_network_simplex`, plus:
- `n_jobs::Int=-1`: Number of parallel jobs (-1 for all threads)
- `verbose::Int=0`: Verbosity level
- `print_every::Int=100`: Print progress every N pairs
- `throw_on_error::Bool=true`: Whether to throw on EMD computation errors

# Returns
- `dist_matrix::Matrix{Float64}`: Symmetric EMD distance matrix
"""
function emds_network_simplex(events0::Vector{<:AbstractMatrix},
             events1::Union{Vector{<:AbstractMatrix},Nothing}=nothing;
             R::Real=1.0, beta::Real=1.0, norm::Bool=false,
             gdim::Int=2, mask::Bool=false,
             n_jobs::Int=-1, print_every::Int=100, verbose::Int=0,
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

# The emd_network_simplex function continues below...

# The emds_network_simplex function continues below...

end # module