using Pkg

function _hepmc3_project_root()
    hepmc3_src = Base.find_package("HepMC3")
    hepmc3_src === nothing && error("HepMC3 is not installed in the active environment")
    return dirname(dirname(hepmc3_src))
end

hepmc3_root = _hepmc3_project_root()
mktempdir() do temp_project
    run(`$(Base.julia_cmd()) --project=$temp_project -e 'using Pkg; Pkg.develop(PackageSpec(path=raw"$hepmc3_root")); Pkg.instantiate()'`)
    run(`$(Base.julia_cmd()) --project=$temp_project $hepmc3_root/gen/build.jl`)
end