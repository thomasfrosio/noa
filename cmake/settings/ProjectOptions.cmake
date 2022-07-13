# Every single project-specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be set from the command line or
# the cmake-gui. Options should be prefixed with `-D` when passed through the command line.

# =====================================================================================
# CPU backend
# =====================================================================================
option(NOA_ENABLE_CPU "\
Build the CPU backend. The CPU backend must be enabled if the unified API is enabled or if tests are built" ON)
option(NOA_ENABLE_OPENMP "Try to enable multithreading, using OpenMP, on the CPU backend" ON)

# BLAS:
option(NOA_ENABLE_BLAS "Try to link and use a BLAS library. See CMake BLA_VENDOR to restrict the search" ON)
option(NOA_BLAS_STATIC "Use the BLAS static libraries instead of the shared ones. Defaults to OFF" OFF)

# FFTW3:
option(NOA_FFTW_THREADS "\
Use a FFTW3 multi-threaded libraries. \
If NOA_ENABLE_OPENMP is ON, the OpenMP version is used instead of the system threads" ON)
option(NOA_FFTW_STATIC "\
Use the FFTW static libraries instead of the shared ones. Defaults to OFF. \
The path search can be restricted using the environmental variables \
NOA_ENV_FFTW_LIBRARIES and NOA_ENV_FFTW_INCLUDE" OFF)

# =====================================================================================
# CUDA backend
# =====================================================================================
option(NOA_ENABLE_CUDA "Try to build the CUDA backend. Requires the CUDA Toolkit. See CMAKE_CUDA_COMPILER" ON)
set(NOA_CUDA_ARCH "" CACHE STRING "\
Architectures to generate device code for (see CUDA_ARCHITECTURES). \
By default, the library tries to use the architecture(s) of the device(s) on the system")
option(NOA_CUDA_CUDART_STATIC "Use the cuda runtime static library instead of the shared ones" ON)
option(NOA_CUDA_CUFFT_STATIC "Use the cuFFT static library instead of the shared ones" OFF)
option(NOA_CUDA_CURAND_STATIC "Use the cuRAND static library instead of the shared ones" OFF)
option(NOA_CUDA_CUBLAS_STATIC "Use the cuBLAS static library instead of the shared ones" OFF)
option(NOA_CUDA_CUSOLVER_STATIC "Use the cuSOLVER static library instead of the shared ones" OFF)

# =====================================================================================
# Unified API
# =====================================================================================
option(NOA_ENABLE_UNIFIED "Build the unified interface. The CPU backend must be enabled" ON)

# =====================================================================================
# Others
# =====================================================================================
option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(NOA_ENABLE_CHECKS_RELEASE "Whether the parameter checks in the unified API should be enabled" ON)

option(NOA_ENABLE_TIFF "Enable support for the TIFF file format. Requires libtiff" ON)
option(NOA_TIFF_STATIC "\
Use the TIFF static library instead of the shared one.
The path search can be restricted using the environmental variables \
NOA_ENV_TIFF_LIBRARIES and NOA_ENV_TIFF_INCLUDE" OFF)

# =====================================================================================
# Targets
# =====================================================================================
option(NOA_BUILD_TESTS "\
Build noa_tests. The `noa::noa_tests` target will be available to the project. When the library is not being imported \
into another project, it defaults to `ON`. Otherwise, defaults to `OFF`" ${NOA_IS_MASTER})
option(NOA_BUILD_BENCHMARKS "\
Build noa_benchmarks. The `noa::noa_benchmarks` target will be made available to the project" ${NOA_IS_MASTER})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
