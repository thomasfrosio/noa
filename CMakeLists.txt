cmake_minimum_required(VERSION 3.15)

# ---------------------------------------------------------------------------------------
# Noa project: the noa library (src/noa) and the higher level programs (src/programs)
# ---------------------------------------------------------------------------------------
project(Noa
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/ffyr2w/noa
        LANGUAGES CXX)

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

# ---------------------------------------------------------------------------------------
# Project wide setup
# ---------------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(${PROJECT_SOURCE_DIR}/cmake/PreventInSourceBuilds.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Options.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Settings.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)

# Usage requirements
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
set_compiler_warnings(project_options)

# ---------------------------------------------------------------------------------------
# Main builds
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/Dependencies.cmake)

add_subdirectory(src/noa)
add_subdirectory(src/programs)

# -----------------------------------------------------------------------------------
# Doxygen - Sphinx documentation
# -----------------------------------------------------------------------------------
if (NOA_BUILD_DOC)
    message(STATUS "Building the documentation")
    add_subdirectory(doc)
else ()
    message(STATUS "Skipping the build of the documentation")
endif ()

# -----------------------------------------------------------------------------------
# Tests - Catch2 and CTest
# -----------------------------------------------------------------------------------
if (NOA_BUILD_TESTS)
    message(STATUS "Building the tests")
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping the build of the tests")
endif ()

# -----------------------------------------------------------------------------------
# Packaging with CPack
# -----------------------------------------------------------------------------------
#if (NOA_PACKAGING)
#    add_subdirectory(packaging)
#endif ()

