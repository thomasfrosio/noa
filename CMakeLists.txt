cmake_minimum_required(VERSION 3.18)

project(Noa
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
include(${PROJECT_SOURCE_DIR}/cmake/PreventInSourceBuilds.cmake)

# ---------------------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/ProjectOptions.cmake)

if (NOA_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
else ()
    message(FATAL_ERROR "The only currently supported GPU backend is CUDA. NOA_BUILD_CUDA should be ON")
endif ()

if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR "The project can only build a static library. Turn OFF BUILD_SHARED_LIBS")
endif ()

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/ProjectEnvironment.cmake)

# These interfaces sets the base project requirements.
add_library(prj_common_option INTERFACE)
add_library(prj_cxx_warnings INTERFACE)

# Make sure everything in the project use C++17.
target_compile_features(prj_common_option INTERFACE cxx_std_17)

# Add the C++ warnings to the C++ compiler.
include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
set_cxx_compiler_warnings(prj_cxx_warnings)

# Go fetch the necessary external contents and make them available to the project.
include(FetchContent)
include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)
# include(${PROJECT_SOURCE_DIR}/ext/tiff/tiff.cmake)
include(${PROJECT_SOURCE_DIR}/ext/fftw/fftw.cmake)

# ---------------------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------------------
# The library: noa
add_subdirectory(src/noa)

# The test framework (using Catch2): noa_tests
if (NOA_BUILD_TESTS)
    message(STATUS "Building the tests")
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping the build of the tests")
endif ()

# The benchmark framework (also using Catch2): noa_benchmarks
if (NOA_BUILD_BENCHMARKS)
    message(STATUS "Building the benchmarks")
    add_subdirectory(benchmarks)
else ()
    message(STATUS "Skipping the build of the benchmarks")
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
