# EnergyFlow.jl EMD Examples

This directory contains examples demonstrating the Earth Mover's Distance (EMD) functionality in EnergyFlow.jl.

## Examples

### 1. Basic EMD Examples (`basic_emd.jl`)

Demonstrates fundamental EMD usage:
- Simple two-event EMD calculation with exact Python comparison
- EMD with weight normalization
- Different parameter settings (R, beta)
- Flow matrix visualization
- Periodic phi handling
- EMD matrix for multiple events
- Spherical distance measure

Run with:
```bash
julia --project=. basic_emd.jl
```

### 2. Negative Weights Workflow (`neg_weights.jl`)

Complete workflow for handling events with negative weights in particle physics:
- Loading ROOT files with particle data
- Applying kinematic cuts (pT > 0.1 GeV, |y| < 4.9)
- Separating particles by production stage (hard process, showered, hadronization)
- Computing EMD between negative-weighted events and all events
- Cell-based reweighting to handle negative weights
- Visualization of results

**Multi-threading support:**
```bash
# Run with automatic thread detection
julia -t auto neg_weights.jl

# Or specify thread count
julia -t 12 neg_weights.jl
```

Features:
- Handles empty events gracefully
- Parallel EMD computation for performance
- Progress tracking
- Comprehensive output statistics

## Data Requirements

The negative weights example requires data files in the `data/` subdirectory:
- `ppzjj_NLO_10k.root` - ROOT file with 10,000 particle physics events
- `event_weights.txt` - Text file containing event weight information
- `ppzjj_NLO_10M_*.npy` - NumPy arrays with 10M event data for comparison:
  - `ppzjj_NLO_10M_weight.npy` - Event weights
  - `ppzjj_NLO_10M_ht.npy` - HT values
  - `ppzjj_NLO_10M_zcoords.npy` - Z boson coordinates
- `hadronization_dist2neg.csv` - Precomputed EMD distances for hadronization
- `showered_dist2neg.csv` - Precomputed EMD distances for showered particles

To right now, to make sure the example work, you can download these files from [Rishabh's Cernbox](https://cernbox.cern.ch/index.php/s/3JqYHjvYJb4m7oL).
In Future, we will provide smaller sample files directly in this repo.

## Notes

- The examples use the exact LP solver (HiGHS) for optimal accuracy (Could be improved with Sinkhorn or other solvers)
- Multi-threading significantly improves performance for EMD calculations (For my Mac with 12 threads, it improved from 7+ hours to around 40 minutes)

## Troubleshooting

If you get an error about EnergyFlow not being found:
1. Make sure you have the correct Julia environment activated. 
2. Make sure you have downloaded the data folder from Rishabh's Cernbox
3. Run `Pkg.instantiate()` to ensure dependencies are installed
4. Verify the EnergyFlow.jl package is properly set up

For optimal performance with `neg_weights.jl`:
- Use Julia with multiple threads: `julia -t auto`
- Ensure sufficient memory for large EMD matrix computations