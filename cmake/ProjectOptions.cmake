# Every single options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be set from
# the command line or e.g. the cmake-gui. Options should be prefixed with `-D` when passed through
# the command line, e.g. cmake -DNOA_ENABLE_CUDA=OFF.
# Don't forget to remove the generated CMakeCache.txt to change options from previous builds.

macro(noa_set_options)
    # CPU backend:
    option(NOA_ENABLE_CPU "Build the CPU backend" ON)
    option(NOA_CPU_OPENMP "Enable multithreading, using OpenMP" ON)
    option(NOA_CPU_MULTITHREADED_FFTW3 "Use the multi-threaded FFTW3 libraries" ${NOA_CPU_OPENMP})

    # CUDA backend:
    option(NOA_ENABLE_CUDA "Build the CUDA backend. Requires a CUDA compiler and a CUDA Toolkit" ON)
    option(NOA_CUDA_STATIC "Link all of the necessary CUDA libraries statically" OFF)

    # Core:
    set(NOA_ERROR_POLICY 2 CACHE STRING "Abort=0, Terminate=1, Exceptions=2")
    option(NOA_ENABLE_WARNINGS "Enable compiler warnings (these only affect the library's source files)" ON)
    option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
    option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" OFF)

    # TIFF:
    option(NOA_ENABLE_TIFF "Enable support for the TIFF file format. Requires static libtiff" OFF) # TODO not tested

    # Additional targets:
    option(NOA_BUILD_TESTS "Build tests" OFF)
    option(NOA_BUILD_BENCHMARKS "Build benchmarks" OFF)
endmacro()
