# ---------------------------------------------------------------------------------------
# Available options
# ---------------------------------------------------------------------------------------

# These *_ENABLE_* options are project wide (i.e. for all targets).
option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_CCACHE "Enable ccache if available" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time" ON)
option(NOA_ENABLE_SINGLE_PRECISION "Use single precision floating-points whenever possible" ON)

#option(BUILD_SHARED_LIBS "Build the library as a shared library." OFF)
option(NOA_BUILD_CUDA "Use the CUDA GPU backend" ON)
set(NOA_CUDA_ARCH 52 60 61 75 85 CACHE STRING "List of architectures to generate device code for. Default=  \"52 60 61 75 85\"" FORCE)
option(NOA_BUILD_OPENCL "Use the OpenCL GPU backend" OFF)
option(NOA_BUILD_APP "Build the executable" ON)
option(NOA_BUILD_TESTS "Build tests" ${NOA_IS_MASTER})
#option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
#option(NOA_PACKAGING "Generate packaging" OFF)
