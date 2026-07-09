# I Don't Know Julia — Just Run It

You have some events and want an EMD number out, without learning Julia first.
Copy-paste your way down this page.

## 1. Install Julia

In a terminal (PowerShell on Windows):

```bash
curl -fsSL https://install.julialang.org | sh    # Windows: winget install julia -s msstore
```

Reopen the terminal, then confirm with `julia --version`.

## 2. Install EnergyFlow (once)

Start Julia by typing `julia`, then at the `julia>` prompt run:

```julia
using Pkg; Pkg.add(["EnergyFlow", "NPZ"])
```

First time only; it takes a few minutes. (`NPZ` reads the `.npz` files below.)

## 3. Run it on your `.npz`

An event is an `M × 3` array: one row per particle, columns `[pT, y, φ]`.

Make a file `my_events.jl` — edit the path and the two indices to pick which
events to compare:

```julia
using EnergyFlow, NPZ

data = npzread("events.npz")          # <-- your file
events = data["events"]               # <-- key holding the event array

# If events is one big padded array of shape (n, maxparticles, 3):
event_A = events[1, :, 1:3]           # <-- first event
event_B = events[2, :, 1:3]           # <-- second event

# norm=true compares shapes only; R sets the distance scale. Leave as-is if unsure.
distance = emd(event_A, event_B; R=1.0, beta=1.0, norm=true)
println("EMD = ", distance)
```

Zero-`pT` padding rows are ignored automatically, so you don't need to trim
them. Run it from the terminal:

```bash
julia my_events.jl        # prints e.g.  EMD = 0.633...
```

Two things to check if the number looks wrong: your array's column order really
is `[pT, y, φ]` (drop or reorder columns as needed), and you picked the right
dict key in `data` — run `keys(data)` to list them.

## What next?

- Tune `R`, `beta`, `norm`: [Getting Started](getting_started.md).
- Compare many events into a distance matrix: [`emds`](@ref).
- Reading HepMC3 files instead: [`load_hepmc3_events`](@ref).
