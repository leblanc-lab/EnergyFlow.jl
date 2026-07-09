# I Don't Know Julia — Just Run It

This page is for people who have their own events and want an EMD number out,
without learning Julia first. Follow it top to bottom, copy-pasting as you go.
You do **not** need to understand the code — you only need to change the parts
this page tells you to change.

## 1. Install Julia

Open a terminal (Terminal on macOS, or a shell on Linux; on Windows use
PowerShell) and run:

```bash
curl -fsSL https://install.julialang.org | sh
```

On Windows, instead run in PowerShell:

```powershell
winget install julia -s msstore
```

This installs [`juliaup`](https://github.com/JuliaLang/juliaup), which manages
Julia for you. Close and reopen your terminal when it finishes, then check it
worked:

```bash
julia --version
```

If that prints a version number (e.g. `julia version 1.11.2`), you're set.

## 2. Install EnergyFlow (once)

Start Julia by typing `julia` and pressing Enter. You'll get a prompt that
looks like `julia>`. Type this line and press Enter:

```julia
using Pkg; Pkg.add("EnergyFlow")
```

This downloads and installs the package. It can take a few minutes the first
time. You only ever have to do this once. When it finishes, type `exit()` to
quit, or just continue to the next step in the same session.

## 3. Put your two events in a file

An **event** is just a table with one row per particle and three numbers per
row:

| pT (or energy) | y (rapidity) | φ (angle) |
|----------------|--------------|-----------|
| 1.0            | 0.5          | 0.1       |
| 0.8            | -0.3         | 0.4       |
| 0.6            | 0.1          | -0.2      |

The first column is "how much" of the particle (its transverse momentum or
energy); the other two are "where" it is. Every event can have a different
number of rows.

Make a plain text file called `my_events.jl` anywhere you like (e.g. on your
Desktop) with the two events you want to compare. Replace the numbers below
with **your** numbers — add or remove rows freely:

```julia
using EnergyFlow

# ---- EDIT THESE TWO TABLES ----
# One row per particle:  pT (or energy)   y   φ
event_A = [ 1.0   0.5   0.1
            0.8  -0.3   0.4
            0.6   0.1  -0.2 ]

event_B = [ 0.9   0.4   0.0
            0.7  -0.2   0.3 ]
# --------------------------------

# R sets the distance scale; norm=true compares only the *shapes*
# (ignores any difference in total pT). Leave these as-is if unsure.
distance = emd(event_A, event_B; R=1.0, beta=1.0, norm=true)

println("EMD = ", distance)
```

Note there are **no commas** between the numbers in a row — just spaces — and a
new line starts a new particle.

## 4. Run it

Back in the terminal, run your file with Julia:

```bash
julia my_events.jl
```

(If the file isn't in the folder your terminal is in, give the full path, e.g.
`julia ~/Desktop/my_events.jl`.)

You'll see a line like:

```
EMD = 0.633...
```

That number is the Energy Mover's Distance between your two events. That's it.

## If your data is in a CSV file

If each event is a CSV file (three columns: pT, y, φ — no header), you can load
them instead of typing the numbers. Add the CSV reader once:

```julia
using Pkg; Pkg.add(["CSV", "DelimitedFiles"])
```

Then a `my_events.jl` that reads two files:

```julia
using EnergyFlow, DelimitedFiles

event_A = readdlm("event_A.csv", ',')   # path to your first file
event_B = readdlm("event_B.csv", ',')   # path to your second file

distance = emd(event_A, event_B; R=1.0, beta=1.0, norm=true)
println("EMD = ", distance)
```

## What next?

- To tune what the number *means* (the `R`, `beta`, and `norm` knobs), see
  [Getting Started](getting_started.md#The-parameters-R,-beta,-and-norm).
- To compare many events at once (a whole distance matrix), see
  [Pairwise EMDs](getting_started.md#Pairwise-EMDs).
- If your events come from a HepMC3 simulation file, you can skip typing them
  in entirely — see
  [Loading events from HepMC3 files](getting_started.md#Loading-events-from-HepMC3-files).
