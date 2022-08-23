# Find the FFTW library (single, double precision versions, with thread support).
# REQUIRE and QUIET are respected, but this module doesn't support versioning.
#
# Paths:
#   The search is done using find_library and find_path. As such, this module looks at the common directories
#   defined by CMake, but it can be guided with the environment variable or CMake variable FFW3_ROOT,
#   or CMAKE_PREFIX_PATH.
#
# Options:
#   The following variables are used:
#       FFTW3_THREADS:  Whether to look for a system-thread version of the fftw3 libraries.
#       FFTW3_OPENMP:   Whether to look for a OpenMP version of the fftw3 libraries.
#       FFTW3_STATIC:   Whether to use the static libraries. Otherwise use the shared ones.
#   Note: FFTW3_OPENMP and FFTW3_THREADS can both be true, but FFTW3_OPENMP takes precedence other FFTW3_THREADS.
#
# The following IMPORTED targets may be created depending on what is found.
# These always include the single and double precision libraries.
# The multi-threaded targets inherits from serial/main library.
# The libraries are considered found if there's at least one "set" that is found.
#   FFTW3::FFTW3:           Serial/main library. Required.
#   FFTW3::FFTW3_threads:   System-threads version. Required if FFTW3_THREADS=ON.
#   FFTW3::FFTW3_openmp:    OpenMP version. Required if FFTW3_OPENMP=ON.
#
# The following variables are set:
#   FFTW3_FOUND
#   FFTW3_THREADS_FOUND
#   FFTW3_OPENMP_FOUND
#   FFTW3_(FLOAT_|DOUBLE_)LIBRARIES
#   FFTW3_(FLOAT_|DOUBLE_)THREADS_LIBRARIES
#   FFTW3_(FLOAT_|DOUBLE_)OPENMP_LIBRARIES
#   FFTW3_INCLUDES
#   FFTW3_INCLUDE_DIRS

# FIXME On Windows, the static and shared libraries have the same file name, including suffix?
set(_old_cmake_lib_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (FFTW3_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else ()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif ()

# Search for the libraries.
find_library(
        FFTW3_DOUBLE_LIBRARIES
        NAMES fftw3 libfftw3-3
)
find_library(
        FFTW3_FLOAT_LIBRARIES
        NAMES fftw3f libfftw3f-3
)
find_library(
        FFTW3_FLOAT_THREADS_LIBRARIES
        NAMES fftw3f_threads
)
find_library(
        FFTW3_DOUBLE_THREADS_LIBRARIES
        NAMES fftw3_threads
)
find_library(
        FFTW3_DOUBLE_OPENMP_LIBRARIES
        NAMES fftw3_omp
)
find_library(
        FFTW3_FLOAT_OPENMP_LIBRARIES
        NAMES fftw3f_omp
)

# Search for the header.
find_path(
        FFTW3_INCLUDES
        NAMES fftw3.h
)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_old_cmake_lib_suffix})

set(FFTW3_LIBRARIES ${FFTW3_FLOAT_LIBRARIES} ${FFTW3_DOUBLE_LIBRARIES})
if (NOT FFT3_FIND_QUIETLY)
    message(STATUS "Found header: ${FFTW3_INCLUDES}")
    message(STATUS "Found single precision: ${FFTW3_FLOAT_LIBRARIES}")
    message(STATUS "Found double precision: ${FFTW3_DOUBLE_LIBRARIES}")
endif ()

# FIXME --with-combined-threads
# The threaded version can be included in the "main" library by using the FFTW3 --with-combined-threads build option.
# This is mainly useful under Windows, to not cary interdependencies. We could check the object file for specific
# symbols only included with thread support, but for now, always require the separate thread library.
# If --with-combined-threads was used, users can turn off the cache variable FFTW3_THREADS and it will work just fine.

# Note that applications can only link against one of the "threaded" version of FFTW3, i.e. system threads or OpenMP.
# Here FFTW3_OPENMP has priority.
set(_fftw3_required_vars FFTW3_FLOAT_LIBRARIES FFTW3_DOUBLE_LIBRARIES)
if (FFTW3_OPENMP)
    if (NOT FFTW3_FLOAT_OPENMP_LIBRARIES OR NOT FFTW3_DOUBLE_OPENMP_LIBRARIES)
        if (FFTW3_FIND_REQUIRED)
            message(STATUS "FFTW3_OPENMP is ON, but could not find the FFTW3-OpenMP libraries")
        endif ()
    else ()
        set(FFTW3_OPENMP_FOUND ON)
        list(APPEND _fftw3_required_vars FFTW3_FLOAT_OPENMP_LIBRARIES FFTW3_DOUBLE_OPENMP_LIBRARIES)
        list(APPEND FFTW3_LIBRARIES ${FFTW3_FLOAT_OPENMP_LIBRARIES} ${FFTW3_DOUBLE_OPENMP_LIBRARIES})

        if (NOT FFT3_FIND_QUIETLY)
            message(STATUS "Found OpenMP single precision: ${FFTW3_FLOAT_OPENMP_LIBRARIES}")
            message(STATUS "Found OpenMP double precision: ${FFTW3_DOUBLE_OPENMP_LIBRARIES}")
        endif ()
    endif ()
elseif (FFTW3_THREADS)
    if (NOT FFTW3_FLOAT_THREADS_LIBRARIES OR NOT FFTW3_DOUBLE_THREADS_LIBRARIES)
        if (FFTW3_FIND_REQUIRED)
            message(FATAL_ERROR "FFTW3_THREADS is ON, but could not find the FFTW3-OpenMP libraries")
        endif ()
    else ()
        set(FFTW3_THREADS_FOUND ON)
        list(APPEND _fftw3_required_vars FFTW3_FLOAT_THREADS_LIBRARIES FFTW3_DOUBLE_THREADS_LIBRARIES)
        list(APPEND FFTW3_LIBRARIES ${FFTW3_FLOAT_THREADS_LIBRARIES} ${FFTW3_DOUBLE_THREADS_LIBRARIES})

        if (NOT FFT3_FIND_QUIETLY)
            message(STATUS "Found threads single precision: ${FFTW3_FLOAT_THREADS_LIBRARIES}")
            message(STATUS "Found threads double precision: ${FFTW3_DOUBLE_THREADS_LIBRARIES}")
        endif ()
    endif ()
endif ()

# Handles REQUIRE and sets FFTW3_FOUND if required variables are valid.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG ${_fftw3_required_vars})

# Targets:
if (FFTW3_FOUND)
    if (FFTW3_STATIC)
        set(_fftw3_lib_type STATIC)
    else ()
        set(_fftw3_lib_type SHARED)
    endif ()

    get_filename_component(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDES} DIRECTORY)
    add_library(FFTW3::FFTW3_float  ${_fftw3_lib_type} IMPORTED)
    add_library(FFTW3::FFTW3_double ${_fftw3_lib_type} IMPORTED)
    set_target_properties(FFTW3::FFTW3_float
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION ${FFTW3_FLOAT_LIBRARIES}
            )
    set_target_properties(FFTW3::FFTW3_double
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION ${FFTW3_DOUBLE_LIBRARIES}
            )

    if (NOT FFT3_FIND_QUIETLY)
        message(STATUS "New imported target available: FFTW3::FFTW3_float")
        message(STATUS "New imported target available: FFTW3::FFTW3_double")
    endif ()

    if (FFTW3_THREADS_FOUND)
        add_library(FFTW3::FFTW3_float_threads ${_fftw3_lib_type} IMPORTED)
        add_library(FFTW3::FFTW3_double_threads ${_fftw3_lib_type} IMPORTED)
        set_target_properties(FFTW3::FFTW3_float_threads
                PROPERTIES
                INTERFACE_LINK_LIBRARIES FFTW3::FFTW3_float
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION ${FFTW3_FLOAT_THREADS_LIBRARIES}
                )
        set_target_properties(FFTW3::FFTW3_double_threads
                PROPERTIES
                INTERFACE_LINK_LIBRARIES FFTW3::FFTW3_double
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION ${FFTW3_DOUBLE_THREADS_LIBRARIES}
                )
        if (NOT FFT3_FIND_QUIETLY)
            message(STATUS "New imported target available: FFTW3::FFTW3_float_threads")
            message(STATUS "New imported target available: FFTW3::FFTW3_double_threads")
        endif ()
    endif ()

    if (FFTW3_OPENMP_FOUND)
        add_library(FFTW3::FFTW3_float_openmp ${_fftw3_lib_type} IMPORTED)
        add_library(FFTW3::FFTW3_double_openmp ${_fftw3_lib_type} IMPORTED)
        set_target_properties(FFTW3::FFTW3_float_openmp
                PROPERTIES
                INTERFACE_LINK_LIBRARIES FFTW3::FFTW3_float
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION ${FFTW3_FLOAT_OPENMP_LIBRARIES}
                )
        set_target_properties(FFTW3::FFTW3_double_openmp
                PROPERTIES
                INTERFACE_LINK_LIBRARIES FFTW3::FFTW3_double
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION ${FFTW3_DOUBLE_OPENMP_LIBRARIES}
                )
        if (NOT FFT3_FIND_QUIETLY)
            message(STATUS "New imported target available: FFTW3::FFTW3_float_openmp")
            message(STATUS "New imported target available: FFTW3::FFTW3_double_openmp")
        endif ()
    endif ()
endif ()
