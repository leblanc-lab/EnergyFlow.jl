# Environment capture for EnergyFlow.jl benchmarks (Julia side).
#
# Every benchmark script includes this file and calls `write_env_block(io)` when
# writing its result table, so that each set of timings records the machine, OS,
# software versions, thread configuration, and source revision it was produced
# on. Without that, a timing table is not reproducible: the reader cannot tell
# whether a difference comes from the code or from the hardware it ran on.
#
# `env_pairs()` returns the environment as (field, value) pairs;
# `write_env_block(io)` renders them as a markdown table.

using Pkg
using Dates

# Every probe below is best-effort: a benchmark must never fail because some
# piece of environment metadata was unavailable (no git, unusual platform, …).
function _probe(f, fallback = "unknown")
    try
        v = f()
        v === nothing && return fallback
        s = strip(string(v))
        isempty(s) ? fallback : s
    catch
        fallback
    end
end

_repo_root() = normpath(joinpath(@__DIR__, ".."))

function _cpu_model()
    _probe() do
        info = Sys.cpu_info()
        isempty(info) ? nothing : strip(info[1].model)
    end
end

function _git_describe()
    commit = _probe() do
        readchomp(Cmd(`git rev-parse --short HEAD`; dir = _repo_root()))
    end
    # "dirty" flags a working tree with uncommitted changes, so a timing can
    # never be silently attributed to a commit that does not contain the code
    # that produced it.
    dirty = _probe(() -> isempty(readchomp(Cmd(`git status --porcelain`; dir = _repo_root()))) ?
                         "clean" : "dirty", "")
    return dirty == "dirty" ? commit * " (uncommitted changes)" : commit
end

function _pkg_version(name::AbstractString)
    _probe() do
        for (_, info) in Pkg.dependencies()
            info.name == name && return info.version
        end
        nothing
    end
end

"""
    env_pairs() -> Vector{Pair{String,String}}

Environment description for the current Julia process. SLURM fields are
included only when running under a scheduler, so laptop runs stay uncluttered.
"""
function env_pairs()
    pairs = [
        "Date"          => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "CPU"           => _cpu_model(),
        "CPU threads"   => string(Sys.CPU_THREADS),
        "Julia threads" => string(Threads.nthreads()),
        "OS"            => string(Sys.KERNEL, " (", Sys.MACHINE, ")"),
        "Julia"         => string(VERSION),
        "EnergyFlow"    => _pkg_version("EnergyFlow"),
        "Commit"        => _git_describe(),
    ]
    # Present only under SLURM; on OSCAR these identify the worker node so a
    # timing can be traced back to the exact allocation that produced it.
    for (var, label) in ("SLURM_JOB_ID"      => "SLURM job",
                         "SLURM_JOB_NODELIST" => "SLURM node",
                         "SLURM_CPUS_ON_NODE" => "SLURM CPUs")
        haskey(ENV, var) && push!(pairs, label => ENV[var])
    end
    return pairs
end

"""
    write_env_block(io)

Write the environment as a markdown table. Called by every benchmark script
immediately after the result-file title.
"""
function write_env_block(io::IO)
    println(io, "## Environment\n")
    println(io, "| Field | Value |")
    println(io, "|---|---|")
    for (k, v) in env_pairs()
        println(io, "| $k | $v |")
    end
    println(io)
end

"""
    print_env()

Echo the environment to stdout at the start of a benchmark run, so a captured
job log carries the same provenance as the result file.
"""
function print_env()
    for (k, v) in env_pairs()
        println(rpad(k, 14), " ", v)
    end
    println()
end
