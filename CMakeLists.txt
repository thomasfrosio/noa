cmake_minimum_required(VERSION 3.18)

project(noa
        LANGUAGES CXX
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/ffyr2w/noa
        )

# The library only supports static linking:
if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR "Dynamic linking is not supported. Turn BUILD_SHARED_LIBS off")
endif ()

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
    if (NOT NOA_CUDA_ARCH)
        include(cmake/settings/CUDASelectArch.cmake)
    endif ()
endif ()

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
include(cmake/settings/ProjectEnvironment.cmake)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_common_option INTERFACE)
add_library(prj_compiler_warnings INTERFACE)

# Make sure everything in the project use C++17.
target_compile_features(prj_common_option INTERFACE cxx_std_17)

# Add the warnings to the compiler(s).
include(cmake/settings/CompilerWarnings.cmake)
set_cxx_compiler_warnings(prj_compiler_warnings)
if (NOA_ENABLE_CUDA)
    set_cuda_compiler_warnings(prj_compiler_warnings)
endif ()

# ---------------------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------------------
add_subdirectory(src/noa)

if (NOA_BUILD_TESTS)
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping target: noa::noa_tests")
endif ()

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
