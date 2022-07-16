# Every single project-specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be set from the command line or
# the cmake-gui. Options should be prefixed with `-D` when passed through the command line.

# To add search locations: CMAKE_PREFIX_PATH

# =====================================================================================
# CPU backend
# =====================================================================================
option(NOA_ENABLE_CPU "Build the CPU backend" ON)
option(NOA_ENABLE_OPENMP "Try to enable multithreading, using OpenMP, on the CPU backend" ON)

# CBLAS and LAPACKE:
# For now, we only support OpenBLAS and require that it was built with CBLAS and LAPACKE.
# Use CMAKE_PREFIX_PATH to the OpenBLAS directory where the CMake config files are installed.

# FFTW
#option(NOA_ENABLE_FFTW "Try to link and use a FFTW library." ON)
option(NOA_FFTW_THREADS "Use a FFTW3 multi-threaded libraries. If OPENMP is enables, the OpenMP version is used" ON)
option(NOA_FFTW_STATIC "Use the FFTW static libraries" OFF)

# =====================================================================================
# CUDA backend
# =====================================================================================
option(NOA_ENABLE_CUDA "Try to build the CUDA backend. Requires the CUDA Toolkit. See CMAKE_CUDA_COMPILER" ON)
set(NOA_CUDA_ARCH "" CACHE STRING "Architectures to generate device code for (see CMake CUDA_ARCHITECTURES)")
option(NOA_CUDA_CUDART_STATIC "Use the cuda runtime static library" ON)
option(NOA_CUDA_CUFFT_STATIC "Use the cuFFT static library" OFF)
option(NOA_CUDA_CURAND_STATIC "Use the cuRAND static library" OFF)
option(NOA_CUDA_CUBLAS_STATIC "Use the cuBLAS static library" OFF)
option(NOA_CUDA_CUSOLVER_STATIC "Use the cuSOLVER static library" OFF)

# =====================================================================================
# Unified API
# =====================================================================================
option(NOA_ENABLE_UNIFIED "Build the unified interface" ON)

# =====================================================================================
# Others
# =====================================================================================
option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(NOA_ENABLE_CHECKS_RELEASE "Whether the parameter checks in the unified API in Release mode should be enabled" ON)

# TIFF
option(NOA_ENABLE_TIFF "Enable support for the TIFF file format. Requires libtiff" ON)
option(NOA_TIFF_STATIC "Use the TIFF static library" OFF)

# =====================================================================================
# Targets
# =====================================================================================
option(NOA_BUILD_TESTS "Build tests" ${NOA_IS_TOP_LEVEL})
option(NOA_BUILD_BENCHMARKS "Build benchmarks" ${NOA_IS_TOP_LEVEL})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
