cmake_minimum_required(VERSION 3.18)

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

if (NOA_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
else ()
    message(FATAL_ERROR "The only currently supported GPU backend is CUDA")
endif ()

if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR "Building the core as a shared library is currently not supported")
    #    message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/Settings.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)

# These interface are used to set host and device options.
add_library(prj_common_option INTERFACE)
target_compile_features(prj_common_option INTERFACE cxx_std_17)

add_library(prj_cxx_warnings INTERFACE)
set_cxx_compiler_warnings(prj_cxx_warnings)

include(FetchContent)
include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)
# include(${PROJECT_SOURCE_DIR}/ext/tiff/tiff.cmake)

# ---------------------------------------------------------------------------------------
# Build it
# ---------------------------------------------------------------------------------------
add_subdirectory(src/noa)

# -----------------------------------------------------------------------------------
# Doxygen - Sphinx documentation
# -----------------------------------------------------------------------------------
#if (NOA_BUILD_DOC)
#    message(STATUS "Building the documentation")
#    add_subdirectory(doc)
#else ()
#    message(STATUS "Skipping the build of the documentation")
#endif ()

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

