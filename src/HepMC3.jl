"""
Minimal utilities for reading
[HepMC3](https://doi.org/10.1016/j.cpc.2020.107310) ASCII files.

Most users should call `EnergyFlow.load_hepmc3_events` for `[pT, η, φ]`
matrices or `EnergyFlow.load_hepmc3_momenta` for `[E, px, py, pz]` matrices.
The lower-level `read_events` interface passes vectors of `Particle` objects to
a callback.
"""
module HepMC3

using LorentzVectorHEP

"""
    Particle{T}

Particle record parsed from a HepMC3 ASCII `P` line.

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

"""
    read_events(f, fin; maxevents=-1, skipevents=0)

Read HepMC3 ASCII events from the open stream `fin`, calling `f(particles)`
once for each selected event. `particles` is a reused `Vector{Particle}`;
copy it inside the callback if it must outlive that callback.

## Arguments
- `maxevents=-1`: maximum number of callbacks; a negative value reads all
  remaining events.
- `skipevents=0`: number of events to skip at the start of the stream.

## Example
```julia
events = Vector{Vector{EnergyFlow.HepMC3.Particle{Float64}}}()
open("events.hepmc") do io
    EnergyFlow.HepMC3.read_events(io; maxevents=10, skipevents=2) do particles
        push!(events, copy(particles))
    end
end
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
`[E, px, py, pz]`, suitable for spherical isotropy. `maxevents=-1` reads all
events and `status=1` keeps final-state particles.
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
