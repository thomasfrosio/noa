# Project options.
# These are CACHE variables (i.e. they are not updated if already set), so they can be set from the command line.
# Using:
#   - NOA_IS_MASTER

# ---------------------------------------------------------------------------------------
# General options
# ---------------------------------------------------------------------------------------

option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_IPO "Enable Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(NOA_ENABLE_PROFILER "Enable the Noa::Profiler" OFF)

# ---------------------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------------------

# CUDA:
option(NOA_BUILD_CUDA "Use the CUDA GPU backend" ON)
set(NOA_CUDA_ARCH
        52 60 61 75 86
        CACHE STRING "List of architectures to generate device code for. Default=\"52 60 61 75 85\""
        FORCE)

# FFTW (see FindFFTW.cmake for more details):
option(NOA_FFTW_USE_STATIC "Use the FFTW static libraries" OFF)
option(NOA_FFTW_USE_OWN "Use your own libraries. If false, fetch them from the web" ON)

# TIFF:
#option(NOA_TIFF_USE_STATIC "Use the TIFF static libraries" ON)
#option(NOA_TIFF_USE_OWN "Use your own libraries. If OFF, CMake will fetch the recommended version from the web" OFF)

# ---------------------------------------------------------------------------------------
# Project target options
# ---------------------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build the noa library as a shared library." OFF)
option(NOA_BUILD_TESTS "Build noa_tests" ${NOA_IS_MASTER})
option(NOA_BUILD_BENCHMARKS "Build noa_benchmarks" ${NOA_IS_MASTER})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
