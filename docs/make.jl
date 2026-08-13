using Documenter
using EnergyFlow

DocMeta.setdocmeta!(EnergyFlow, :DocTestSetup, :(using EnergyFlow); recursive=true)

makedocs(
    modules=[EnergyFlow],
    sitename="EnergyFlow.jl",
    checkdocs=:exports,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://leblanc-lab.github.io/EnergyFlow.jl/stable",
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Just Run It (No Julia Experience)" => "man/just_run_it.md",
            "Getting Started" => "man/getting_started.md",
            "Tutorial" => "man/tutorial.md",
            "Backends and Metrics" => "man/backends_and_metrics.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(
    repo="github.com/leblanc-lab/EnergyFlow.jl.git",
    devbranch="main",
)