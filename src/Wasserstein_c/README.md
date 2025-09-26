# Wasserstein C++ Library Headers

This directory contains only the essential C++ headers from the Wasserstein library needed for the CxxWrap Julia interface.

## Directory Structure

```
Wasserstein_c/
├── boost/          # Required Boost headers (histogram, mp11, etc.)
└── src/
    └── wasserstein/
        ├── Wasserstein.hh   # Main library header
        └── internal/        # Internal implementation headers
```

## Removed Files

To keep the package lightweight, we removed:
- Python bindings and SWIG interface
- Documentation and examples
- Test files and benchmarks
- Build configuration files (CMakeLists.txt, pyproject.toml, etc.)
- CI/CD workflows (.github)
- OpenMP parallel implementations

## Usage

These headers are used by the C++ wrapper in `src/cxxwrap/`. The wrapper provides:
- `emd_cxx()` - Earth Mover's Distance computation
- `emds_cxx()` - Pairwise EMD matrix computation

See `src/cxxwrap/README.md` for build instructions.

## License

The Wasserstein library is originally from: https://github.com/pkomiske/Wasserstein
Original license terms apply to these header files.