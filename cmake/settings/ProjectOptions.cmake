# Every single project-specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be
# set from the command line or the cmake-gui.

# =====================================================================================
# General options
# =====================================================================================

option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(NOA_ENABLE_PROFILER "Enable the noa::Profiler" OFF)

# =====================================================================================
# Backend and Dependencies
# =====================================================================================

# CUDA
# ====
option(NOA_ENABLE_CUDA "Build the CUDA backend" ON)
set(NOA_CUDA_ARCH 61 CACHE STRING "Architectures to generate device code for. Default=61")
option(NOA_CUDA_USE_CUFFT_STATIC "Use the cuFFT static library instead of the shared ones" OFF)

# CPU
# ===
option(NOA_ENABLE_CPU "Build the CPU backend. The CPU backend must be enabled if the unified interface is enable or if tests are built" ON)
option(NOA_ENABLE_OPENMP "Enable multithreading, using OpenMP, on the CPU backend" ON)

# FFTW (see noa/ext/fftw for more details):
option(NOA_FFTW_USE_EXISTING "Enable support for the FFTW3 Fast Fourier transforms" ON)
option(NOA_FFTW_USE_EXISTING "Use the installed FFTW3 libraries. If OFF, the libraries are fetched from the web" ON)
option(NOA_FFTW_USE_STATIC "Use the FFTW static libraries instead of the shared ones" OFF)

# COMMON
# ======

# TIFF (see noa/ext/tiff for more details):
option(NOA_ENABLE_TIFF "Enable support for the TIFF file format" ON)
option(NOA_TIFF_USE_EXISTING "Use the installed TIFF library. If OFF, the library is fetched from the web" ON)
option(NOA_TIFF_USE_STATIC "Use the TIFF static library instead of the shared one" OFF)

# UNIFIED
# =======
option(NOA_ENABLE_UNIFIED "Build the unified interface. The CPU backend must be enabled" ON)

# =====================================================================================
# Project target options
# =====================================================================================
option(NOA_BUILD_TESTS "Build noa_tests" ${NOA_IS_MASTER})
option(NOA_BUILD_BENCHMARKS "Build noa_benchmarks" ${NOA_IS_MASTER})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
