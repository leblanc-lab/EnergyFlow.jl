# Negative Weights Reweighting Workflow - Julia Script
# This is a Julia version of the Python notebook "Neg weights workshop.ipynb"

using Pkg
Pkg.activate("../..")
using EnergyFlow
using UnROOT
using Plots
using LinearAlgebra
using Statistics
using ProgressMeter
using NPZ
using DelimitedFiles
using StatsBase
using Base.Threads

println("Running with $(Threads.nthreads()) threads")
println("Number of CPU cores: $(Sys.CPU_THREADS)")
println("To use multiple threads, run with: julia -t auto neg_weights.jl")

# Read ROOT file
println("Reading ROOT file...")
file = ROOTFile("data/ppzjj_NLO_10k.root")
tree = LazyTree(file, "Events", ["Particle_energy", "Particle_px","Particle_py", "Particle_pz", 
                                 "Particle_status", "Particle_pid", "Particle_d1", "Particle_d2", "Particle_mass"])

# Define kinematic cut parameters
particle_y_cut = 4.9
particle_pt_cut = 0.1
N = 10000

println("Available keys: ", keys(tree[1]))

# Look at event 1
println("\nEvent 1 analysis:")
println("Number of particles: ", length(tree[1].Particle_mass))
println("Highest mass particle: ", maximum(tree[1].Particle_mass), " GeV")

# Find max particles after cuts
println("\nFinding maximum particles after cuts...")
max_particles = 0

for i in 1:N
    # Get particle data for this event
    n_particles = length(tree[i].Particle_energy)
    if n_particles == 0
        continue
    end
    
    # Create four-momentum matrix
    four_momenta = zeros(n_particles, 4)
    four_momenta[:, 1] = tree[i].Particle_energy
    four_momenta[:, 2] = tree[i].Particle_px
    four_momenta[:, 3] = tree[i].Particle_py
    four_momenta[:, 4] = tree[i].Particle_pz
    
    # Apply cuts using EnergyFlow
    cut_momenta = EnergyFlow.apply_kinematic_cuts(four_momenta, 
                                                  pT_min=particle_pt_cut, 
                                                  y_max=particle_y_cut, 
                                                  coords="cartesian")
    
    n_kept = size(cut_momenta, 1)
    if n_kept > max_particles
        global max_particles = n_kept
    end
end

println("Maximum particles after cuts: $max_particles")

# Process all events
println("\nProcessing all events...")
main_coords = zeros(N, max_particles, 3)
p_stat = 4 * ones(N, max_particles)
main_pid = zeros(N, max_particles)
main_daughter1 = zeros(N, max_particles)
main_daughter2 = zeros(N, max_particles)

@showprogress "Processing events..." for i in 1:N
    n_particles = length(tree[i].Particle_energy)
    if n_particles == 0
        continue
    end
    
    # Create four-momentum matrix
    four_momenta = zeros(n_particles, 4)
    four_momenta[:, 1] = tree[i].Particle_energy
    four_momenta[:, 2] = tree[i].Particle_px
    four_momenta[:, 3] = tree[i].Particle_py
    four_momenta[:, 4] = tree[i].Particle_pz
    
    # Apply cuts
    cut_momenta = EnergyFlow.apply_kinematic_cuts(four_momenta, 
                                                  pT_min=particle_pt_cut, 
                                                  y_max=particle_y_cut, 
                                                  coords="cartesian")
    
    n_kept = size(cut_momenta, 1)
    if n_kept > 0
        # Transform to hadronic coordinates
        hadronic = EnergyFlow.cartesian_to_hadronic(cut_momenta)
        main_coords[i, 1:n_kept, :] = hadronic
        
        # Find which particles were kept
        kept_indices = Int[]
        for j in 1:n_particles
            pT = sqrt(four_momenta[j, 2]^2 + four_momenta[j, 3]^2)
            y = abs(atanh(four_momenta[j, 4] / four_momenta[j, 1]))
            if pT >= particle_pt_cut && y <= particle_y_cut
                push!(kept_indices, j)
            end
        end
        
        if length(kept_indices) > 0
            p_stat[i, 1:length(kept_indices)] = tree[i].Particle_status[kept_indices]
            main_pid[i, 1:length(kept_indices)] = tree[i].Particle_pid[kept_indices]
            main_daughter1[i, 1:length(kept_indices)] = tree[i].Particle_d1[kept_indices]
            main_daughter2[i, 1:length(kept_indices)] = tree[i].Particle_d2[kept_indices]
        end
    end
end

# Remove empty events
zero_array = Int[]
for i in 1:N
    if all(main_coords[i, :, :] .== 0)
        push!(zero_array, i)
    end
end

if !isempty(zero_array)
    keep_events = trues(N)
    keep_events[zero_array] .= false
    
    main_coords = main_coords[keep_events, :, :]
    p_stat = p_stat[keep_events, :]
    main_pid = main_pid[keep_events, :]
    main_daughter1 = main_daughter1[keep_events, :]
    main_daughter2 = main_daughter2[keep_events, :]
end

println("Removed $(length(zero_array)) empty events")
N_filtered = size(main_coords, 1)

# Extract pT and eta-phi
EMD_pt = main_coords[:, :, 1]
EMD_etaphi = main_coords[:, :, 2:3]

# Separate particles by type
println("\nSeparating particles by type...")
hadronization_EMD_pt = zeros(N_filtered, max_particles)
hardprocess_EMD_pt = zeros(N_filtered, max_particles)
showered_EMD_pt = zeros(N_filtered, max_particles)
hadronization_EMD_etaphi = zeros(N_filtered, max_particles, 2)
hardprocess_EMD_etaphi = zeros(N_filtered, max_particles, 2)
showered_EMD_etaphi = zeros(N_filtered, max_particles, 2)

# Track which events have no particles of each type
empty_hardprocess = Int[]
empty_showered = Int[]
empty_hadronization = Int[]

for i in 1:N_filtered
    # Hadronization particles
    fs_indices = findall((main_daughter1[i, :] .== -1) .& (main_daughter2[i, :] .== -1))
    if !isempty(fs_indices)
        temp_fs = length(fs_indices)
        hadronization_EMD_pt[i, 1:temp_fs] = EMD_pt[i, fs_indices]
        hadronization_EMD_etaphi[i, 1:temp_fs, :] = EMD_etaphi[i, fs_indices, :]
    else
        push!(empty_hadronization, i)
    end
    
    # Hard process particles
    sp_indices = findall(p_stat[i, :] .== 23)
    if !isempty(sp_indices)
        temp_sp = length(sp_indices)
        hardprocess_EMD_pt[i, 1:temp_sp] = EMD_pt[i, sp_indices]
        hardprocess_EMD_etaphi[i, 1:temp_sp, :] = EMD_etaphi[i, sp_indices, :]
    else
        push!(empty_hardprocess, i)
    end
    
    # Showered particles
    sh_indices = findall((p_stat[i, :] .> 70) .& (p_stat[i, :] .< 80))
    if !isempty(sh_indices)
        temp_sh = length(sh_indices)
        showered_EMD_pt[i, 1:temp_sh] = EMD_pt[i, sh_indices]
        showered_EMD_etaphi[i, 1:temp_sh, :] = EMD_etaphi[i, sh_indices, :]
    else
        push!(empty_showered, i)
    end
end

println("Events with no hard process particles: $(length(empty_hardprocess))")
println("Events with no showered particles: $(length(empty_showered))")
println("Events with no hadronization particles: $(length(empty_hadronization))")

# Trim arrays
max1 = max_particles
max2 = max_particles
max3 = max_particles

for i in 1:N_filtered
    A = count(hadronization_EMD_pt[i, :] .== 0)
    if A < max1
        global max1 = A
    end
    
    B = count(hardprocess_EMD_pt[i, :] .== 0)
    if B < max2
        global max2 = B
    end
        
    C = count(showered_EMD_pt[i, :] .== 0)
    if C < max3
        global max3 = C
    end
end

hadronization_EMD_pt = hadronization_EMD_pt[:, 1:(max_particles - max1)]
hardprocess_EMD_pt = hardprocess_EMD_pt[:, 1:(max_particles - max2)]
showered_EMD_pt = showered_EMD_pt[:, 1:(max_particles - max3)]

hadronization_EMD_etaphi = hadronization_EMD_etaphi[:, 1:(max_particles - max1), :]
hardprocess_EMD_etaphi = hardprocess_EMD_etaphi[:, 1:(max_particles - max2), :]
showered_EMD_etaphi = showered_EMD_etaphi[:, 1:(max_particles - max3), :]

println("\nMax particles per type:")
println("Hard process: $(max_particles - max2)")
println("Showered: $(max_particles - max3)")
println("Hadronization: $(max_particles - max1)")

# Read event weights
println("\nReading event weights...")
weights_file = "data/event_weights.txt"
event_weights = Float64[]

open(weights_file, "r") do file
    for line in eachline(file)
        if occursin("weight:", line)
            weight_str = split(line, "weight: ")[2]
            weight = parse(Float64, weight_str)
            push!(event_weights, weight)
        end
    end
end

if length(event_weights) > N_filtered
    event_weights = event_weights[1:N_filtered]
end

# Apply weight scaling
weight = 1.455e4
event_weights *= weight

# Find negative events
neg_events = Int[]
for i in 1:length(event_weights)
    if event_weights[i] < 0
        push!(neg_events, i)
    end
end

println("Found $(length(neg_events)) negative weighted events")

# Load 10M weights
big_event_weight = npzread("data/ppzjj_NLO_10M_weight.npy")
rescale = sum(big_event_weight) / sum(event_weights)
event_weights *= rescale
big_event_weight *= weight

println("Rescaling done successfully: ", abs(1 - sum(big_event_weight) / sum(event_weights)) < 1e-5)

# Prepare EMD calculation function
function prepare_events_for_emd(pt_array, etaphi_array, event_idx)
    mask = pt_array[event_idx, :] .> 0
    pt = pt_array[event_idx, mask]
    etaphi = etaphi_array[event_idx, mask, :]
    
    n_particles = length(pt)
    if n_particles == 0
        return zeros(0, 3)
    end
    
    event = zeros(n_particles, 3)
    event[:, 1] = pt
    event[:, 2:3] = etaphi
    
    return event
end

# Compute EMD distances for hard process particles
println("\nComputing EMD distances for hard process particles...")
hardprocess_dist2neg = zeros(length(neg_events), N_filtered)

# Filter out negative events that have no hard process particles
valid_neg_events_hard = [i for (idx, i) in enumerate(neg_events) if !(i in empty_hardprocess)]
println("Valid negative events for hard process: $(length(valid_neg_events_hard)) out of $(length(neg_events))")

if Threads.nthreads() > 1
    println("Using $(Threads.nthreads()) threads for EMD computation...")
    
    # Create thread-safe progress tracking
    emd_progress_counter = Threads.Atomic{Int}(0)
    total_computations = length(neg_events)
    emd_progress = Progress(total_computations, desc="Hard process EMD...")
    
    # Parallel computation over negative events
    @threads for (i_idx, i) in collect(enumerate(neg_events))
        # Skip if this negative event has no hard process particles
        if i in empty_hardprocess
            hardprocess_dist2neg[i_idx, :] .= 1000.0
            Threads.atomic_add!(emd_progress_counter, 1)
            ProgressMeter.update!(emd_progress, emd_progress_counter[])
            continue
        end
        
        event_i = prepare_events_for_emd(hardprocess_EMD_pt, hardprocess_EMD_etaphi, i)
        
        for j in 1:N_filtered
            # Skip if target event has no hard process particles
            if j in empty_hardprocess
                hardprocess_dist2neg[i_idx, j] = 1000.0
                continue
            end
            
            event_j = prepare_events_for_emd(hardprocess_EMD_pt, hardprocess_EMD_etaphi, j)
            
            # Compute EMD
            try
                hardprocess_dist2neg[i_idx, j] = emd(event_i, event_j, R=1.0, beta=1.0, norm=false)
            catch e
                # This should not happen now that we're checking for empty events
                println("Thread $(Threads.threadid()): Unexpected error between events $i and $j: $e")
                hardprocess_dist2neg[i_idx, j] = 1000.0
            end
        end
        
        Threads.atomic_add!(emd_progress_counter, 1)
        ProgressMeter.update!(emd_progress, emd_progress_counter[])
    end
else
    # Single-threaded version
    @showprogress "Hard process EMD..." for (i_idx, i) in enumerate(neg_events)
        # Skip if this negative event has no hard process particles
        if i in empty_hardprocess
            hardprocess_dist2neg[i_idx, :] .= 1000.0
            continue
        end
        
        event_i = prepare_events_for_emd(hardprocess_EMD_pt, hardprocess_EMD_etaphi, i)
        
        for j in 1:N_filtered
            # Skip if target event has no hard process particles
            if j in empty_hardprocess
                hardprocess_dist2neg[i_idx, j] = 1000.0
                continue
            end
            
            event_j = prepare_events_for_emd(hardprocess_EMD_pt, hardprocess_EMD_etaphi, j)
            
            # Compute EMD
            try
                hardprocess_dist2neg[i_idx, j] = emd(event_i, event_j, R=1.0, beta=1.0, norm=false)
            catch e
                # This should not happen now that we're checking for empty events
                println("Unexpected error between events $i and $j: $e")
                hardprocess_dist2neg[i_idx, j] = 1000.0
            end
        end
    end
end

# Create proximity matrix
hardprocess_close2neg = zeros(Int, length(neg_events), N_filtered)
for i in 1:length(neg_events)
    hardprocess_close2neg[i, :] = sortperm(hardprocess_dist2neg[i, :])
end
hardprocess_close2neg = hardprocess_close2neg[:, 2:end]

println("\nEMD computation completed!")
println("First few distances for event $(neg_events[1]): ", hardprocess_dist2neg[1, 1:5])

# Load precomputed matrices for showered/hadronized
println("\nLoading precomputed distance matrices...")
if isfile("data/hadronization_dist2neg.csv") && isfile("data/showered_dist2neg.csv")
    hadronization_dist2neg = readdlm("data/hadronization_dist2neg.csv", ',')
    showered_dist2neg = readdlm("data/showered_dist2neg.csv", ',')
    println("Loaded precomputed matrices successfully")
else
    println("Warning: Precomputed matrices not found!")
    if Threads.nthreads() > 1
        println("Consider computing showered/hadronization EMD with multi-threading as well.")
    end
end

println("\nWorkflow completed!")
if Threads.nthreads() == 1
    println("Tip: Run with julia -t auto neg_weights.jl to use multiple threads for faster computation")
end