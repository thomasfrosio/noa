# Every single project specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be
# set from the command line or the cmake-gui.

# =====================================================================================
# General options
# =====================================================================================

option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_LTO "Enable Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(NOA_ENABLE_PROFILER "Enable the noa::Profiler" OFF)

# =====================================================================================
# Dependencies
# =====================================================================================

# CUDA:
option(NOA_BUILD_CUDA "Use the CUDA GPU backend" ON)
set(NOA_CUDA_ARCH
        52 60 61 75 86
        CACHE STRING "List of architectures to generate device code for. Default=\"52 60 61 75 85\""
        FORCE)

# FFTW (see noa/ext/fftw.cmake for more details):
option(NOA_FFTW_USE_STATIC "Use or build the FFTW static libraries." OFF)

# ---------------------------------------------------------------------------------------
# Project target options
# ---------------------------------------------------------------------------------------
option(NOA_BUILD_TESTS "Build noa_tests" ${NOA_IS_MASTER})
option(NOA_BUILD_BENCHMARKS "Build noa_benchmarks" ${NOA_IS_MASTER})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
