cmake_minimum_required(VERSION 3.14)
project(WassersteinWrapper)

# Find Julia and JlCxx
find_package(JlCxx REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find OpenMP, but we'll manually set it if not found
find_package(OpenMP QUIET)

# Manually configure OpenMP for macOS with Homebrew
if(NOT OpenMP_CXX_FOUND AND APPLE)
    # Check common Homebrew locations
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_FOUND TRUE)
        message(STATUS "Manually configured OpenMP from Homebrew (ARM)")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIRS "/usr/local/opt/libomp/include")
        set(OpenMP_CXX_FOUND TRUE)
        message(STATUS "Manually configured OpenMP from Homebrew (Intel)")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c/boost
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c  # Add parent dir for boost includes
)

# Add OpenMP include directory if found
if(OpenMP_CXX_FOUND)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# Add the shared library
add_library(wasserstein_wrapper SHARED wasserstein_wrapper.cpp)

# Set compile flags
if(OpenMP_CXX_FOUND)
    set_target_properties(wasserstein_wrapper PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(wasserstein_wrapper JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl ${OpenMP_omp_LIBRARY})
    message(STATUS "OpenMP enabled - building with parallel support")
    message(STATUS "  OpenMP flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "  OpenMP library: ${OpenMP_omp_LIBRARY}")
else()
    target_link_libraries(wasserstein_wrapper JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)
    message(STATUS "OpenMP not found - building without parallel support")
endif()

# Set properties
set_property(TARGET wasserstein_wrapper PROPERTY POSITION_INDEPENDENT_CODE ON)

# Install the library
install(TARGETS wasserstein_wrapper
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../)