# Find the FFTW library (single, double precision versions, with thread support).
# This should work both on Linux and Windows.
#
# The following variables will be used:
#   NOA_FFTW_THREADS:               Looks for a multithreaded version of the fftw3 libraries.
#   NOA_ENABLE_OPENMP:              If true, prioritize the multithreaded version using OpenMP.
#   NOA_FFTW_STATIC:                If true, only static libraries are found, otherwise both static and shared.
#   (ENV) NOA_ENV_FFTW_LIBRARIES:   If set and not empty, the libraries are exclusively searched under this path.
#   (ENV) NOA_ENV_FFTW_INCLUDE:     If set and not empty, the headers (i.e. fftw3.h) are exclusively searched under this path.
#
# The following targets are created (containing both single-threaded and optionally the multi-threaded libraries):
#   fftw3::float
#   fftw3::double
#
# The following variables are set:
#   NOA_FFTW_FLOAT_FOUND
#   NOA_FFTW_DOUBLE_FOUND

# Log input variables:
message(STATUS "[input] NOA_FFTW3_STATIC: ${NOA_FFTW3_STATIC}")
message(STATUS "[input (env)] NOA_ENV_FFTW3_LIBRARIES: $ENV{NOA_ENV_FFTW3_LIBRARIES}")
message(STATUS "[input (env)] NOA_ENV_FFTW3_INCLUDE: $ENV{NOA_ENV_FFTW3_INCLUDE}")

# Whether to search for static or dynamic libraries.
set(NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (NOA_FFTW_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

if (DEFINED ENV{NOA_ENV_FFTW_LIBRARIES} AND NOT $ENV{NOA_ENV_FFTW_LIBRARIES} STREQUAL "")
    find_library(
            NOA_FFTW_DOUBLE_LIB
            NAMES "fftw3" libfftw3-3
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
            REQUIRED
    )
    find_library(
            NOA_FFTW_FLOAT_LIB
            NAMES "fftw3f" libfftw3f-3
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
            REQUIRED
    )

    find_library(
            NOA_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_DOUBLE_OPENMP_LIB
            NAMES "fftw3_omp"
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_FLOAT_OPENMP_LIB
            NAMES "fftw3f_omp"
            PATHS $ENV{NOA_ENV_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
else ()
    find_library(
            NOA_FFTW_DOUBLE_LIB
            NAMES "fftw3"
            PATHS ${LIB_INSTALL_DIR}
            REQUIRED
    )
    find_library(
            NOA_FFTW_FLOAT_LIB
            NAMES "fftw3f"
            PATHS ${LIB_INSTALL_DIR}
            REQUIRED
    )

    find_library(
            NOA_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_DOUBLE_OPENMP_LIB
            NAMES "fftw3_omp"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_FLOAT_OPENMP_LIB
            NAMES "fftw3f_omp"
            PATHS ${LIB_INSTALL_DIR}
    )
endif ()

if (DEFINED ENV{NOA_ENV_FFTW_INCLUDE} AND NOT $ENV{NOA_ENV_FFTW_INCLUDE} STREQUAL "")
    find_path(NOA_FFTW_INC
            NAMES "fftw3.h"
            PATHS $ENV{NOA_ENV_FFTW_INCLUDE}
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH
            REQUIRED
            )
else ()
    find_path(NOA_FFTW_INC
            NAMES "fftw3.h"
            PATHS ${INCLUDE_INSTALL_DIR}
            REQUIRED
            )
endif ()

# Reset to whatever it was:
set(CMAKE_FIND_LIBRARY_SUFFIXES ${NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD})

# Add multithreaded libraries:
message(STATUS "[output] NOA_FFTW_DOUBLE_LIB: ${NOA_FFTW_DOUBLE_LIB}")
message(STATUS "[output] NOA_FFTW_FLOAT_LIB: ${NOA_FFTW_FLOAT_LIB}")
if (NOA_FFTW_THREADS)
    if (NOA_ENABLE_OPENMP)
        if (NOA_FFTW_FLOAT_OPENMP_LIB)
            message(STATUS "[output] NOA_FFTW_FLOAT_OPENMP_LIB: ${NOA_FFTW_FLOAT_OPENMP_LIB}")
            set(NOA_FFTW_FLOAT_LIB ${NOA_FFTW_FLOAT_LIB} ${NOA_FFTW_FLOAT_OPENMP_LIB})
        else ()
            message(FATAL_ERROR "With NOA_FFTW_THREADS=1, the OpenMP versions of fftw3 are required but could not be found")
        endif ()
        if (NOA_FFTW_DOUBLE_OPENMP_LIB)
            message(STATUS "[output] NOA_FFTW_DOUBLE_OPENMP_LIB: ${NOA_FFTW_DOUBLE_OPENMP_LIB}")
            set(NOA_FFTW_DOUBLE_LIB ${NOA_FFTW_DOUBLE_LIB} ${NOA_FFTW_DOUBLE_OPENMP_LIB})
        else ()
            message(FATAL_ERROR "With NOA_FFTW_THREADS=1, the OpenMP versions of fftw3 are required but could not be found")
        endif ()
    else ()
        if (NOA_FFTW_FLOAT_THREADS_LIB)
            message(STATUS "[output] NOA_FFTW_FLOAT_THREADS_LIB: ${NOA_FFTW_FLOAT_THREADS_LIB}")
            set(NOA_FFTW_FLOAT_LIB ${NOA_FFTW_FLOAT_LIB} ${NOA_FFTW_FLOAT_THREADS_LIB})
        endif ()
        if (NOA_FFTW_DOUBLE_THREADS_LIB)
            message(STATUS "[output] NOA_FFTW_DOUBLE_THREADS_LIB: ${NOA_FFTW_DOUBLE_THREADS_LIB}")
            set(NOA_FFTW_DOUBLE_LIB ${NOA_FFTW_DOUBLE_LIB} ${NOA_FFTW_DOUBLE_THREADS_LIB})
        endif ()
    endif ()
endif ()

# Targets:
set(NOA_FFTW_FLOAT_LIB_FOUND TRUE)
add_library(fftw3::float INTERFACE IMPORTED)
set_target_properties(fftw3::float
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_FLOAT_LIB}"
        )

set(NOA_FFTW_DOUBLE_LIB_FOUND TRUE)
add_library(fftw3::double INTERFACE IMPORTED)
set_target_properties(fftw3::double
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_DOUBLE_LIB}"
        )

message(STATUS "New imported target available: fftw3::float")
message(STATUS "New imported target available: fftw3::double")

mark_as_advanced(
        NOA_FFTW_DOUBLE_LIB
        NOA_FFTW_FLOAT_LIB
        NOA_FFTW_DOUBLE_THREADS_LIB
        NOA_FFTW_FLOAT_THREADS_LIB
        NOA_FFTW_DOUBLE_OPENMP_LIB
        NOA_FFTW_FLOAT_OPENMP_LIB
        NOA_FFTW_INC
)
