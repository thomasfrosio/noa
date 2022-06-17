message(STATUS "FFTW3: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

# FIXME: fftw3-omp is weird
# If multithreading is required, we can use either the library using the system threads or the OpenMP threads.
# However, the OMP version is taking a LOT of time for some transforms... which doesn't seem right...
# For now, use the system threads.
if (NOA_FFTW_THREADS AND NOA_ENABLE_OPENMP)
    set(NOA_ENABLE_OPENMP_OLD ${NOA_ENABLE_OPENMP})
    set(NOA_ENABLE_OPENMP OFF CACHE INTERNAL "OpenMP for FFTW3")
endif ()

find_package(FFTW)

# FIXME: fftw3-omp is weird
# Restore the old value
set(NOA_ENABLE_OPENMP ${NOA_ENABLE_OPENMP_OLD} CACHE BOOL "Enable multithreading, using OpenMP, on the CPU backend" FORCE)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "FFTW3: searching for existing libraries... done")

# Note: FetchContent is not very practical with non-CMake projects.
#       FFTW added CMake support in 3.3.7 but it seems to be experimental even in 3.3.8.
# TODO: Add support for FetchContent or ExternalProject_Add.
