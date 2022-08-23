cmake_minimum_required(VERSION 3.18)

project(noa
        LANGUAGES CXX
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/ffyr2w/noa
        )

# Friendly heads up. The library only supports static linking:
if (BUILD_SHARED_LIBS)
    message(STATUS "BUILD_SHARED_LIBS is ON, but dynamic linking is not supported. This will be ignored.")
endif ()

# Whether this is the top-level project
if (NOT DEFINED NOA_IS_TOP_LEVEL)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(NOA_IS_TOP_LEVEL ON)
    else ()
        set(NOA_IS_TOP_LEVEL OFF)
    endif ()
endif ()

# Add the local modules.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Make sure different configurations don't collide.
set(CMAKE_DEBUG_POSTFIX "d")

# Generate compile_commands.json to make it easier to work with clang based tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------------------
include(cmake/util/PreventInSourceBuilds.cmake)
include(cmake/util/ProjectOptions.cmake)

# Check for CUDA support and select the GPU architecture if user didn't provide one.
if (NOA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        if (NOA_CUDA_FIND_ARCHITECTURE)
            include(${PROJECT_SOURCE_DIR}/cmake/util/CUDASelectArch.cmake)
            set(CMAKE_CUDA_ARCHITECTURES ${CUDASelectArch_RESULTS})
        endif ()

    else ()
        message(FATAL_ERROR "No CUDA support detected, so building without CUDA")
    endif ()
endif ()

include(cmake/util/ProjectToolchain.cmake)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_common_option INTERFACE)
add_library(prj_compiler_warnings INTERFACE)
target_compile_features(prj_common_option INTERFACE cxx_std_17)
target_compile_options(prj_common_option INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

# Add the warnings to the compiler(s).
include(cmake/util/CompilerWarnings.cmake)
set_cxx_compiler_warnings(prj_compiler_warnings)
set_cuda_compiler_warnings(prj_compiler_warnings)

# CMAKE_CURRENT_BINARY_DIR is the build directory and for top level projects this is where we call CMake from.
# This is also where we put the generated files:
set(NOA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated")
set(NOA_GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/noa_generated_headers")

# CMAKE_INSTALL_PREFIX is the path of the install directory, which is often specified in the command line.
# Then, we need the paths (relative to CMAKE_INSTALL_PREFIX) to know where to install things.
# Get the standard install directories. This should be included before creating the targets since they'll
# use these paths to define their install include directories.
include(GNUInstallDirs)
# In our case, we'll use:
# CMAKE_INSTALL_BINDIR (bin/):           Executables, scripts and symlinks intended for end users to run directly.
# CMAKE_INSTALL_LIBDIR (lib/):           Libraries and object files.
# CMAKE_INSTALL_INCLUDEDIR (include/):   Header files.

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

# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------
include(cmake/util/ProjectExport.cmake)
