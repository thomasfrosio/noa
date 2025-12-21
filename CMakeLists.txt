cmake_minimum_required(VERSION 3.30)

# Extract version from header file before project() call.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/noa/Version.hpp" _noa_version_header)
if (NOT _noa_version_header MATCHES "NOA_VERSION ([0-9]+)([0-9][0-9])([0-9][0-9])")
    message(FATAL_ERROR "Cannot get NOA_VERSION from Version.hpp.")
endif ()
math(EXPR NOA_VERSION_MAJOR ${CMAKE_MATCH_1})
math(EXPR NOA_VERSION_MINOR ${CMAKE_MATCH_2})
math(EXPR NOA_VERSION_PATCH ${CMAKE_MATCH_3})
set(NOA_VERSION ${NOA_VERSION_MAJOR}.${NOA_VERSION_MINOR}.${NOA_VERSION_PATCH})
message(STATUS "NOA_VERSION=${NOA_VERSION}")

# Create the project.
project(noa VERSION ${NOA_VERSION} LANGUAGES CXX)

# Set project-specific options before other configurations.
include(cmake/ProjectOptions.cmake)
noa_set_options()

# Modern CMake policies.
cmake_policy(SET CMP0077 NEW)  # option() honors normal variables

# Dynamic linking is not supported.
if(BUILD_SHARED_LIBS)
    message(WARNING "BUILD_SHARED_LIBS=ON but noa does not support dynamic linking. Building as static library.")
    set(BUILD_SHARED_LIBS OFF)
endif()

if(PROJECT_IS_TOP_LEVEL)
    # Prevent in-source builds.
    get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)
    if ("${_srcdir}" STREQUAL "${_bindir}")
        message("In-source builds are a bad idea")
        message("Please create a separate build directory (which can be inside the project directory) and run cmake from there")
        message(FATAL_ERROR "Quitting configuration")
    endif ()

    # Set default build type if not specified.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
    endif()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json for clang-based tools
    set(CMAKE_DEBUG_POSTFIX "d") # make sure debug and release builds don't collide
endif()

# C++ standard configuration (for dependencies that need it).
if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
elseif (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++20 or later is required, but got CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # use -std=c++20 instead of -std=gnu++20

# Sets and prints info about the toolchain.
include(cmake/ProjectToolchain.cmake)
noa_set_toolchain(NOA_ENABLE_CUDA)

# Compiler warnings and options.
include(cmake/utils/CompilerOptions.cmake)
add_library(prj_compiler_public_options INTERFACE)
add_library(prj_compiler_private_options INTERFACE)
add_library(prj_compiler_warnings INTERFACE)
noa_set_compiler_options(prj_compiler_public_options prj_compiler_private_options)
if(NOA_ENABLE_WARNINGS)
    noa_set_compiler_warnings(prj_compiler_warnings NOA_ENABLE_WARNINGS_AS_ERRORS)
endif()

# Project targets.
add_subdirectory(src)

if (PROJECT_IS_TOP_LEVEL AND NOA_BUILD_TESTS)
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping target: noa::noa_tests")
endif ()

if (PROJECT_IS_TOP_LEVEL AND NOA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
else ()
    message(STATUS "Skipping target: noa::noa_benchmarks")
endif ()

# Export the project.
include(cmake/ProjectExport.cmake)
