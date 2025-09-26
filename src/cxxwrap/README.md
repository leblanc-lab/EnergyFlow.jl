# C++ Wasserstein Wrapper for Julia

This directory contains the C++ wrapper for the Wasserstein library using CxxWrap.jl.

## Building the Wrapper

### Prerequisites

1. **CMake** (>= 3.14)
   ```bash
   # macOS
   brew install cmake

   # Ubuntu/Debian
   sudo apt-get install cmake
   ```

2. **C++ Compiler** with C++14 support
   - macOS: Xcode Command Line Tools
   - Linux: gcc/g++ or clang

3. **Julia CxxWrap Package**
   ```julia
   using Pkg
   Pkg.add("CxxWrap")
   ```

### Build Instructions

1. Navigate to this directory:
   ```bash
   cd src/cxxwrap
   ```

2. Run the build script:
   ```bash
   ./build_wrapper.sh
   ```

   Or manually:
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_PREFIX_PATH=$(julia -e 'using CxxWrap; println(CxxWrap.prefix_path())')
   make
   make install
   ```

3. The library will be installed to `src/libwasserstein_wrapper.dylib` (macOS) or `src/libwasserstein_wrapper.so` (Linux).

## Troubleshooting

### Symbol Not Found Error

If you get a "Symbol not found" error when loading the library, it may be due to a version mismatch with CxxWrap. Try:

1. Rebuild CxxWrap:
   ```julia
   using Pkg
   Pkg.build("CxxWrap")
   ```

2. Clean and rebuild the wrapper:
   ```bash
   rm -rf build
   ./build_wrapper.sh
   ```

### Missing Headers

The wrapper requires the Boost histogram headers which are included in `src/Wasserstein_c/boost/`. If you get header not found errors, ensure the Wasserstein_c directory is present.

## Usage

Once built, the C++ wrapper provides:

- `emd_cxx(event1, event2; kwargs...)` - Compute EMD between two events
- `emds_cxx(events; kwargs...)` - Compute pairwise EMD matrix

These functions offer the same interface as the Julia implementations but use the optimized C++ backend.

## Performance

The C++ wrapper typically provides:
- 2-3x speedup over the Julia network simplex implementation
- 10-20x speedup over the HiGHS exact solver
- Native SIMD optimizations for distance calculations