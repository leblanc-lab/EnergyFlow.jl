# Contributing to EnergyFlow.jl

This document describes the best ways for reporting issues, proposing changes, and contributing via pull requests.

By contributing to this project either with code or discussion, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs

Open an [issue](https://github.com/leblanc-lab/EnergyFlow.jl/issues) and include:

- A minimal reproducible example (ideally using `EnergyFlow` alone, without other packages)
- The output of `julia --project -e 'using Pkg; Pkg.status()'`
- What you expected to happen vs. what actually happened

## Suggesting features

Open an issue describing the use case before writing code, especially for
anything that touches the public API (new backends, metrics, or solver
options).

## Development setup

```bash
git clone https://github.com/leblanc-lab/EnergyFlow.jl.git
cd EnergyFlow.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```


### Running tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

For faster iteration on a single test file:

```bash
julia --project test/runtests.jl
```

The CI runs on Julia LTS, stable (`1`), and pre-release on Linux, plus
a thread-safety check (single-threaded precompile, multi-threaded run).

A macOS job runs on PRs labeled `macos`, but not by default.

It should go without saying, but if your change affects solver internals or anything `@threads`-parallel, please check that it doesn't break under multithreaded execution.


### Building the docs

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Generated files are written to `docs/build/`.

## Making a pull request

1. Fork the repo and create a branch off `main`.
2. Make your change, adding or updating tests in [test/](test/) that cover it.
3. Update [docs/src/](docs/src/) if you changed public API behavior, and the
   `README.md` quick-start if it's affected.
4. Make sure `Pkg.test()` passes locally.
5. Open a PR against `main` with a description of what changed and why.

Small, focused PRs are much easier to review than large ones. If your change
is straightforward to split into independent pieces, please consider opening separate PRs.

## Code style

Follow the conventions already used in the surrounding file (naming,
docstrings, argument order). There's no enforced formatter yet, so match the
local style rather than reformatting unrelated code.

A formatter will be added in the future, and we will adopt it for all new code afterwards.

## Questions

If something is unclear, open an issue or reach out to @mattleblanc directly.
