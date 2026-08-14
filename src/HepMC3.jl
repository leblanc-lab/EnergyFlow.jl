"""
Module providing a minimal support to read HepMC3 ascii files.

This module contains functions and types for reading HepMC3 ASCII files. It
provides a `Particle` struct to represent particles in the event, and a
`read_events` function to read events from a file.

## Types
- `Particle{T}`: A struct representing a particle in the event. It contains
  fields for momentum, status, PDG ID, barcode, and vertex.
- `T`: The type parameter for the `Particle` struct, specifying the precision of
  the momentum components.

## Functions
- `read_events(f, fin; maxevents=-1, skipevents=0)`: Reads events from a HepMC3
  ascii file. Each event is passed to the provided function `f` as a vector of
  `Particle`s. The `maxevents` parameter specifies the maximum number of events
  to read (-1 to read all available events), and the `skipevents` parameter
  specifies the number of events to skip at the beginning of the file.

    Example usage:
    ```julia
    function process_event(particles)
        # Process the event
    end

    file = open("events.hepmc", "r")
    read_events(process_event, file, maxevents=10, skipevents=2)
    close(file)
    ```

    - `f`: The function to be called for each event. It should take a single
      argument of type `Vector{Particle}`.
    - `fin`: The input file stream to read from.
    - `maxevents`: The maximum number of events to read. Default is -1, which
      reads all available events.
    - `skipevents`: The number of events to skip at the beginning of the file.
      Default is 0.

## References
- [HepMC3](https://doi.org/10.1016/j.cpc.2020.107310): The HepMC3 software
  package.
"""
module HepMC3

using LorentzVectorHEP

"""
    struct Particle{T}

A struct representing a particle in the HepMC3 library.

# Fields
- `momentum::LorentzVector{T}`: The momentum of the particle.
- `status::Int`: The status code of the particle.
- `pdgid::Int`: The PDG ID of the particle.
- `barcode::Int`: The barcode of the particle.
- `vertex::Int`: The vertex ID of the particle.

"""
struct Particle{T}
    momentum::LorentzVector{T}
    status::Int
    pdgid::Int
    barcode::Int
    vertex::Int
end

Particle{T}() where {T} = Particle(LorentzVector{T}(0.0, 0.0, 0.0, 0.0), 0, 0, 0, 0)

""" Read a [HepMC3](https://doi.org/10.1016/j.cpc.2020.107310) ascii file.

    Each event is passed to the provided function f as a vector of Particles. A
    maximum number of events to read (value -1 to read all available events) and
    a number of events to skip at the beginning of the file can be provided.
"""

"""
    read_events(f, fin; maxevents=-1, skipevents=0)

Reads events from a file and processes them.

## Arguments
- `f`: A function that takes an array of `Particle{T}` as input and processes
  it.
- `fin`: The input file object.
- `maxevents`: The maximum number of events to read. Default is -1, which means
  all events will be read.
- `skipevents`: The number of events to skip before processing. Default is 0.

## Description
This function reads events from a file and processes them using the provided
function `f`. Each event is represented by a series of lines in the file, where
each line corresponds to a particle. The function parses the lines and creates
`Particle{T}` objects, which are then passed to the processing function `f`. The
function `f` can perform any desired operations on the particles.

## Example
```julia
f = open(fname)
events = Vector{PseudoJet}[]
HepMC3.read_events(f, maxevents = maxevents, skipevents = skipevents) do parts
    input_particles = PseudoJet[]
    for p in parts
        if p.status == 1
            push!(
                input_particles,
                PseudoJet(p.momentum.x, p.momentum.y, p.momentum.z, p.momentum.t),
            )
        end
    end
    push!(events, input_particles)
end
close(f)
```
"""
function read_events(f, fin; maxevents = -1, skipevents = 0)
    T = Float64
    particles = Particle{T}[]
    toskip = skipevents
    emitted = 0
    in_event = false

    for l in eachline(fin)
        if occursin(r"HepMC::.*-END_EVENT_LISTING", l)
            break
        end

        tok = split(l)
        isempty(tok) && continue

        if tok[1] == "E"
            # Starting a new event means the previous one is complete
            if in_event
                if toskip > 0
                    toskip -= 1
                elseif maxevents < 0 || emitted < maxevents
                    f(particles)
                    emitted += 1
                else
                    break
                end
            end
            empty!(particles)
            in_event = true

        elseif tok[1] == "P" && in_event
            # Skip particle parsing for skipped events or after maxevents
            if toskip > 0 || (maxevents >= 0 && emitted >= maxevents)
                continue
            end

            length(tok) < 10 && continue

            barcode = parse(Int, tok[2])
            vertex = parse(Int, tok[3])
            pdgid = parse(Int, tok[4])
            px = parse(T, tok[5])
            py = parse(T, tok[6])
            pz = parse(T, tok[7])
            e = parse(T, tok[8])
            status = parse(Int, tok[10])
            push!(particles,
                  Particle{T}(LorentzVector(e, px, py, pz), status, pdgid, barcode, vertex))
        end
    end

    # Process the last event if present.
    if in_event
        if toskip > 0
            return
        end
        if maxevents < 0 || emitted < maxevents
            f(particles)
        end
    end
end

"""
    load_hepmc3_momenta(filepath; maxevents=-1, status=1) -> Vector{Matrix{Float64}}

Load events from a HepMC3 ASCII file as `M×4` matrices with columns
`[E, px, py, pz]`, suitable for spherical isotropy.
"""
function load_hepmc3_momenta(filepath::AbstractString; maxevents::Int=-1, status::Int=1)
    events = Matrix{Float64}[]
    raw = NTuple{4,Float64}[]
    n_ev = 0
    open(filepath) do f
        for line in eachline(f)
            occursin(r"HepMC::.*-END_EVENT_LISTING", line) && break
            c = isempty(line) ? '\0' : line[1]
            if c == 'E' && length(line) > 1 && line[2] == ' '
                if !isempty(raw)
                    push!(events, reduce(vcat, (reshape(collect(r), 1, 4) for r in raw)))
                    n_ev += 1
                end
                empty!(raw)
                (maxevents >= 0 && n_ev >= maxevents) && break
            elseif c == 'P' && length(line) > 1 && line[2] == ' '
                tok = split(line)
                length(tok) < 10 && continue
                parse(Int, tok[10]) == status || continue
                px = parse(Float64, tok[5]); py = parse(Float64, tok[6])
                pz = parse(Float64, tok[7]); e = parse(Float64, tok[8])
                push!(raw, (e, px, py, pz))
            end
        end
        if !isempty(raw) && (maxevents < 0 || n_ev < maxevents)
            push!(events, reduce(vcat, (reshape(collect(r), 1, 4) for r in raw)))
        end
    end
    return events
end

end
