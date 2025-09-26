#!/bin/bash

# Build script for Wasserstein C++ wrapper

echo "Building Wasserstein C++ wrapper for Julia..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create build directory
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find Julia CxxWrap prefix path
JULIA_PREFIX=$(julia -e 'using CxxWrap; println(CxxWrap.prefix_path())' 2>/dev/null)

if [ -z "$JULIA_PREFIX" ]; then
    echo "Error: CxxWrap not found. Install it with: julia -e 'using Pkg; Pkg.add(\"CxxWrap\")'"
    exit 1
fi

echo "Julia CxxWrap prefix: $JULIA_PREFIX"

# Configure with cmake
echo "Configuring with cmake..."
cmake .. \
    -DCMAKE_PREFIX_PATH="$JULIA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-std=c++14 -fPIC"

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

# Install to parent src directory
echo "Installing..."
make install

if [ $? -ne 0 ]; then
    echo "Installation failed"
    exit 1
fi

echo "Build completed successfully!"
echo "Library installed to: $SCRIPT_DIR/../"