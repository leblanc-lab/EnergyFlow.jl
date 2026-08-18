# Contributing to EnergyFlow.jl

This document describes how to report issues, propose changes, and contribute
pull requests.

By contributing code or participating in project discussions, you agree to
abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs

Open an [issue](https://github.com/leblanc-lab/EnergyFlow.jl/issues) and include:

- A minimal reproducible example (ideally using `EnergyFlow` alone)
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

CI runs on Julia LTS, stable (`1`), and pre-release on Linux, plus a
thread-safety check that precompiles with one thread and runs with multiple
threads. The LTS and pre-release jobs are currently allowed to fail; the
stable Julia job is required.

A macOS job runs on PRs labeled `macos`, but not by default.

If your change affects solver internals or parallel code, also run the tests
with multiple threads:

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
```

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

Small, focused PRs are easier to review. If your change splits naturally into
independent pieces, consider opening separate pull requests.

## Code style

Follow the conventions already used in the surrounding file (naming,
docstrings, argument order). There's no enforced formatter yet, so match the
local style rather than reformatting unrelated code.

## Questions

If something is unclear, open a
[GitHub issue](https://github.com/leblanc-lab/EnergyFlow.jl/issues).
