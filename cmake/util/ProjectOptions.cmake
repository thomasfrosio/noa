# Every single project-specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be set from
# the command line or e.g. the cmake-gui. Options should be prefixed with `-D` when passed through
# the command line, e.g. cmake -DNOA_ENABLE_CUDA=OFF.

# =====================================================================================
# CUDA backend
# =====================================================================================
option(NOA_ENABLE_CUDA "Build the CUDA backend. Requires a CUDA compiler and the CUDA Toolkit" ON)
if (NOA_ENABLE_CUDA)
    # Note that a CUDA capable GPU is _not_ required on the system that compiles the library.
    # However, if one or multiple GPU are available, the library will target the architecture of these GPUs by default.
    # Turn this option OFF if this behavior is not desired.
    # To specify the CUDA architecture, use the CMake variable CMAKE_CUDA_ARCHITECTURES.
    option(NOA_CUDA_FIND_ARCHITECTURE "Overwrite CMAKE_CUDA_ARCHITECTURES with the architecture(s) of the host GPU(s)" ON)

    # CUDA Toolkit
    option(NOA_CUDA_CUDART_STATIC "Use the cuda runtime static library" ON)
    option(NOA_CUDA_CUFFT_STATIC "Use the cuFFT static library" OFF)
    option(NOA_CUDA_CURAND_STATIC "Use the cuRAND static library" OFF)
    option(NOA_CUDA_CUBLAS_STATIC "Use the cuBLAS static library" OFF)
    option(NOA_CUDA_CUSOLVER_STATIC "Use the cuSOLVER static library" OFF)
endif ()

# =====================================================================================
# CPU backend
# =====================================================================================
option(NOA_ENABLE_CPU "Build the CPU backend" ON)
if (NOA_ENABLE_CPU)
    option(NOA_CPU_OPENMP "Enable multithreading, using OpenMP" ON)

    set(NOA_CPU_MATH_LIBRARY "CBLAS-LAPACKE-FFTW3" CACHE STRING "Maths libraries for the CPU backend. Should be CBLAS-LAPACKE-FFTW3 or MKL")
    if (NOA_CPU_MATH_LIBRARY STREQUAL "CBLAS-LAPACKE-FFTW3")
        option(BLA_STATIC "Whether the use the static BLA/CBLAS library. Otherwise use the shared ones" OFF)
        option(BLA_VENDOR "BLAS vendor. See https://cmake.org/cmake/help/latest/module/FindBLAS.html" "")
        if (${BLA_VENDOR} MATCHES "Intel")
            message(FATAL_ERROR "BLA_VENDOR=${BLA_VENDOR} is not compatible with this mode")
        endif()

        # Only used if LAPACKE is not part of the LAPACK library.
        option(LAPACKE_STATIC "Whether the use the static LAPACKE library. Otherwise use the shared ones" ${BLA_STATIC})

        # FFTW
        option(FFTW3_THREADS "Use the multi-threaded FFTW3 libraries using system threads" OFF)
        option(FFTW3_OPENMP "Use the multi-threaded FFTW3 libraries using OpenMP. Takes precedence over FFTW3_THREADS" ${NOA_CPU_OPENMP})
        option(FFTW3_STATIC "Use the FFTW3 static libraries" OFF)

    elseif (NOA_CPU_MATH_LIBRARY STREQUAL "MKL")
        message(FATAL_ERROR "MKL is not supported yet") # TODO Add MKL support
    else ()
        message(FATAL_ERROR "NOA_CPU_MATH_LIBRARY is not recognized. Should be OpenBLAS or MKL, but got ${NOA_CPU_MATH_LIBRARY}")
    endif ()
endif()

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
option(TIFF_STATIC "Use the TIFF static library instead of the shared ones." OFF)

# =====================================================================================
# Targets
# =====================================================================================
option(NOA_BUILD_TESTS "Build tests" ON)
option(NOA_BUILD_BENCHMARKS "Build benchmarks" OFF)
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
