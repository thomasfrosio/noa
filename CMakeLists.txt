cmake_minimum_required(VERSION 3.24)

project(noa
        LANGUAGES CXX
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/thomasfrosio/noa
        )

# Friendly heads up. The library only supports static linking:
if (BUILD_SHARED_LIBS)
    message(STATUS "BUILD_SHARED_LIBS is ON, but dynamic linking is not supported. This will be ignored for noa.")
endif ()

# Only set the cxx_standard if it is not set by someone else.
# noa doesn't use this, but it is useful to set a default for fetch dependencies.
if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

# Add the local modules.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Make sure debug configuration doesn't collide with release.
set(CMAKE_DEBUG_POSTFIX "d")

# Generate compile_commands.json to make it easier to work with clang based tools.
# This is also helpful to check what cmake is passing to the compilers for build time.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled.
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------------------
include(cmake/utils/PreventInSourceBuilds.cmake)
include(cmake/utils/CompilerOptions.cmake)
include(cmake/ProjectOptions.cmake)
include(cmake/ProjectToolchain.cmake)

noa_prevent_in_source_builds()
noa_set_options()
noa_set_toolchain(NOA_ENABLE_CUDA)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_common_option INTERFACE)
add_library(prj_compiler_warnings INTERFACE)
noa_set_compiler_options(prj_common_option)
if (NOA_ENABLE_WARNINGS)
    noa_set_cxx_compiler_warnings(prj_compiler_warnings NOA_ENABLE_WARNINGS_AS_ERRORS)
    if (NOA_ENABLE_CUDA)
        noa_set_cuda_compiler_warnings(prj_compiler_warnings NOA_ENABLE_WARNINGS_AS_ERRORS)
    endif ()
endif ()

# Save the current binary directory, so that we can easily refer to it even within subdirectories.
set(NOA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

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

#if (NOA_BUILD_BENCHMARKS)
#    add_subdirectory(benchmarks)
#else ()
#    message(STATUS "Skipping target: noa::noa_benchmarks")
#endif ()

# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------
include(cmake/ProjectExport.cmake)
