cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------------------
# Project preamble
# ---------------------------------------------------------------------------------------
# Automatically parse the version number from header file.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/noa/Version.hpp" _noa_version_header)
if (NOT _noa_version_header MATCHES "NOA_VERSION ([0-9]+)([0-9][0-9])([0-9][0-9])")
    message(FATAL_ERROR "Cannot get NOA_VERSION from Version.hpp.")
endif ()
# Use math to skip leading zeros if any.
math(EXPR NOA_VERSION_MAJOR ${CMAKE_MATCH_1})
math(EXPR NOA_VERSION_MINOR ${CMAKE_MATCH_2})
math(EXPR NOA_VERSION_PATCH ${CMAKE_MATCH_3})
set(NOA_VERSION ${NOA_VERSION_MAJOR}.${NOA_VERSION_MINOR}.${NOA_VERSION_PATCH})
message(STATUS "NOA_VERSION=${NOA_VERSION}")

# Allow overriding options in a parent project via `set` before including noa.
cmake_policy(SET CMP0077 NEW)

# Friendly heads up. The library only supports static linking:
if (BUILD_SHARED_LIBS)
    message(STATUS "BUILD_SHARED_LIBS is ON, but dynamic linking is not supported. This will be ignored for noa.")
endif ()

# Only set the C++ standard if it is not set by someone else.
# noa doesn't use this (we use the newer target_compile_features alternative instead),
# but it is useful to set as default for fetched dependencies that depend on it, like fmt.
if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
elseif (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard should be at least 20, got CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")
endif()
set(CMAKE_CXX_EXTENSIONS OFF) # use -std=c++20 instead of e.g. -std=gnu++20

# Make sure debug configuration doesn't collide with release.
set(CMAKE_DEBUG_POSTFIX "d")

# Generate compile_commands.json to make it easier to work with clang based tools.
# This is also helpful to check what cmake is passing to the compilers for build time.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------------------
project(noa LANGUAGES CXX)

include(cmake/utils/PreventInSourceBuilds.cmake)
include(cmake/utils/CompilerOptions.cmake)
include(cmake/ProjectOptions.cmake)
include(cmake/ProjectToolchain.cmake)

noa_prevent_in_source_builds()
noa_set_options()
noa_set_toolchain(NOA_ENABLE_CUDA)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_compiler_public_options INTERFACE)
add_library(prj_compiler_private_options INTERFACE)
add_library(prj_compiler_warnings INTERFACE)
noa_set_compiler_options(prj_compiler_public_options prj_compiler_private_options)
if (NOA_ENABLE_WARNINGS)
    noa_set_compiler_warnings(prj_compiler_warnings NOA_ENABLE_WARNINGS_AS_ERRORS)
endif ()

# CMAKE_INSTALL_PREFIX is the path of the install directory, which is often specified in the command line.
# Then, we need the paths to know where to install things relative to CMAKE_INSTALL_PREFIX.
# Get the standard install directories. This should be included before creating the targets since they'll
# use these paths to define their install include directories.
include(GNUInstallDirs)
# In our case, we'll use:
# CMAKE_INSTALL_BINDIR (bin/)
# CMAKE_INSTALL_LIBDIR (lib/)
# CMAKE_INSTALL_INCLUDEDIR (include/)

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
include(cmake/ProjectExport.cmake)
