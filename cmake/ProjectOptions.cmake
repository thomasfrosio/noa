# Every single options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be set from
# the command line or e.g. the cmake-gui. Options should be prefixed with `-D` when passed through
# the command line, e.g. cmake -DNOA_ENABLE_CUDA=OFF.

macro(noa_set_options)
    # CPU backend:
    option(NOA_ENABLE_CPU "Build the CPU backend" ON)
    if (NOA_ENABLE_CPU)
        option(NOA_CPU_OPENMP "Enable multithreading, using OpenMP" ON)
        option(NOA_MULTITHREADED_FFTW3 "Use the multi-threaded FFTW3 libraries using system threads" ${NOA_CPU_OPENMP})
    endif ()

    # CUDA backend:
    option(NOA_ENABLE_CUDA "Build the CUDA backend. Requires a CUDA compiler and a CUDA Toolkit" ON)
    if (NOA_ENABLE_CUDA)
        option(NOA_CUDA_STATIC "Link all necessary CUDA libraries statically." OFF)
        option(NOA_CUDA_JIT "Whether to use the JIT/online mode (runtime compilation) instead of the offline mode" OFF)
        option(NOA_CUDA_ENABLE_ASSERT "Enable device assertions, which can be useful for debugging but considerably increases the device link time. This only affects Debug builds, since assertions are turned off in Release" OFF)
    endif ()

    # Core:
    option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
    option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
    option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" OFF)
    option(NOA_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
    if (NOA_ENABLE_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT result OUTPUT output)
        if(result)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        else()
            message(SEND_ERROR "IPO is not supported: ${output}")
        endif()
    endif ()

    # TIFF:
    option(NOA_ENABLE_TIFF "Enable support for the TIFF file format. Requires static libtiff" ON)

    # Additional targets:
    option(NOA_BUILD_TESTS "Build tests" OFF)
    option(NOA_BUILD_BENCHMARKS "Build benchmarks" OFF)
endmacro()
