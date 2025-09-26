cmake_minimum_required(VERSION 3.14)
project(WassersteinWrapper)

# Find Julia and JlCxx
find_package(JlCxx REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c/boost
    ${CMAKE_CURRENT_SOURCE_DIR}/../Wasserstein_c  # Add parent dir for boost includes
)

# Add the shared library
add_library(wasserstein_wrapper SHARED wasserstein_wrapper.cpp)

# Link with JlCxx
target_link_libraries(wasserstein_wrapper JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)

# Set properties
set_property(TARGET wasserstein_wrapper PROPERTY POSITION_INDEPENDENT_CODE ON)

# Install the library
install(TARGETS wasserstein_wrapper
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../)