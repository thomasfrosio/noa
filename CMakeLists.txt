cmake_minimum_required(VERSION 3.18)
# CMAKE_CUDA_ARCHITECTURES was added in 3.18

project(noa
        LANGUAGES CXX
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/ffyr2w/noa
        )

# ---------------------------------------------------------------------------------------
# Whether or not this is the top-level (master) project
# ---------------------------------------------------------------------------------------
if (NOT DEFINED NOA_IS_MASTER)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(NOA_IS_MASTER ON)
    else ()
        set(NOA_IS_MASTER OFF)
    endif ()
endif ()
include(cmake/settings/PreventInSourceBuilds.cmake)

# ---------------------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------------------
include(cmake/settings/ProjectOptions.cmake)

if (NOA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
endif ()

if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR "noa is a static library. Turn BUILD_SHARED_LIBS OFF")
endif ()

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
include(cmake/settings/ProjectEnvironment.cmake)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_common_option INTERFACE)
add_library(prj_cxx_warnings INTERFACE)

# Make sure everything in the project use C++17.
target_compile_features(prj_common_option INTERFACE cxx_std_17)

# Add the C++ warnings to the C++ compiler.
include(cmake/settings/CompilerWarnings.cmake)
set_cxx_compiler_warnings(prj_cxx_warnings)

# Go fetch the necessary external contents and make them available to the project.
set(NOA_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)

include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/ext/tiff/tiff.cmake)
include(${PROJECT_SOURCE_DIR}/ext/fftw/fftw.cmake)

# ---------------------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------------------
add_subdirectory(src/noa)

# The test framework (using Catch2): noa_tests
if (NOA_BUILD_TESTS)
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping target: noa::noa_tests")
endif ()

# The benchmark framework (using Catch2): noa_benchmarks
if (NOA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
else ()
    message(STATUS "Skipping target: noa::noa_benchmarks")
endif ()

# The Doxygen-Sphinx documentation
#if (NOA_BUILD_DOC)
#    message(STATUS "Building the documentation")
#    add_subdirectory(doc)
#else ()
#    message(STATUS "Skipping the build of the documentation")
#endif ()

# -----------------------------------------------------------------------------------
# Project packaging
# -----------------------------------------------------------------------------------
#if (NOA_PACKAGING)
#    add_subdirectory(packaging)
#endif ()
