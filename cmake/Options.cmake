# ---------------------------------------------------------------------------------------
# Available options
# ---------------------------------------------------------------------------------------

# These *_ENABLE_* options are project wide (i.e. for all targets).
option(NOA_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(NOA_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NOA_ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(NOA_ENABLE_CCACHE "Enable ccache if available" OFF)
option(NOA_ENABLE_PCH "Build using precompiled header to speed up compilation time" ON)

# The noa, "noa", is a static (or shared) library. The main application, "utopia", links to
# this library privately. If one only wants the library, NOA_BUILD_APP can be turned off.
# If one wants the noa available as a shared library, this is possible as well.
option(NOA_BUILD_APP "Build the executable" ON)
option(BUILD_SHARED_LIBS "Build the library as a shared library." OFF)

# By default, build the tests, the documentation and install + package
# the targets ONLY if this is the top-level project.
option(NOA_BUILD_TESTS "Build tests" ${NOA_IS_MASTER})
option(NOA_BUILD_DOC "Build Doxygen-Sphinx documentation" OFF)
option(NOA_PACKAGING "Generate packaging" OFF)
