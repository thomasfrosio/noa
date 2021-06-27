# Find the FFTW library (single, double precision versions, with thread support.).
# This should work both on Linux and Windows.
#
# The following variables will be used:
#   NOA_FFTW_USE_STATIC:      If true, only static libraries are found, otherwise both static and shared.
#   (ENV) NOA_FFTW_LIBRARIES: If set and not empty, the libraries are exclusively searched under this path.
#   (ENV) NOA_FFTW_INCLUDE:   If set and not empty, the headers (i.e. fftw3.h) are exclusively searched under this path.
#
# The following targets are created:
#   fftw3::float
#   fftw3::double
#   fftw3::float_threads    (can be "empty", since fftw3::float can already include the thread support)
#   fftw3::double_threads   (can be "empty", since fftw3::double can already include the thread support)

# Whether to search for static or dynamic libraries.
set(NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (NOA_FFTW_USE_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

if (DEFINED ENV{NOA_FFTW_LIBRARIES} AND NOT $ENV{NOA_FFTW_LIBRARIES} STREQUAL "")
    find_library(
            NOA_FFTW_FLOAT_LIB
            NAMES "fftw3f" libfftw3f-3
            PATHS $ENV{NOA_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS $ENV{NOA_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_DOUBLE_LIB
            NAMES "fftw3" libfftw3-3
            PATHS $ENV{NOA_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            NOA_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS $ENV{NOA_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
else ()
    find_library(
            NOA_FFTW_DOUBLE_LIB
            NAMES "fftw3"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_FLOAT_LIB
            NAMES "fftw3f"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            NOA_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
endif ()

if (DEFINED ENV{NOA_FFTW_INCLUDE} AND NOT $ENV{NOA_FFTW_INCLUDE} STREQUAL "")
    find_path(NOA_FFTW_INC
            NAMES "fftw3.h"
            PATHS $ENV{NOA_FFTW_INCLUDE}
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH
            )
else ()
    find_path(NOA_FFTW_INC
            NAMES "fftw3.h"
            PATHS ${INCLUDE_INSTALL_DIR}
            )
endif ()

# Reset to whatever it was.
set(CMAKE_FIND_LIBRARY_SUFFIXES ${NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD})

# Logging
if (NOA_FFTW_FLOAT_LIB)
    message(STATUS "NOA_FFTW_FLOAT_LIB: ${NOA_FFTW_FLOAT_LIB}")
else ()
    message(FATAL_ERROR "Could not find fftw3f on the system.")
endif ()

if (NOA_FFTW_DOUBLE_LIB)
    message(STATUS "NOA_FFTW_DOUBLE_LIB: ${NOA_FFTW_DOUBLE_LIB}")
else ()
    message(FATAL_ERROR "Could not find fftw3 on the system.")
endif ()

if (NOA_FFTW_FLOAT_THREADS_LIB)
    message(STATUS "NOA_FFTW_FLOAT_THREADS_LIB: ${NOA_FFTW_FLOAT_THREADS_LIB}")
else ()
    message(WARNING "Could not find fftw3f threads on the system."
            "This is allowed since the \"--with-combined-threads\" option might have been used to build FFTW."
            "In any case, we now assume that the main FFTW library includes the thread support.")
endif ()

if (NOA_FFTW_DOUBLE_THREADS_LIB)
    message(STATUS "NOA_FFTW_DOUBLE_THREADS_LIB: ${NOA_FFTW_DOUBLE_THREADS_LIB}")
else ()
    message(WARNING "Could not find fftw3 threads on the system."
            "This is allowed since the \"--with-combined-threads\" option might have been used to build FFTW."
            "In any case, we now assume that the main FFTW library includes the thread support.")
endif ()

if (NOA_FFTW_INC)
    message(STATUS "NOA_FFTW_INC: ${NOA_FFTW_INC}")
else ()
    message(FATAL_ERROR "Could not find the fftw3.h header on the system.")
endif ()

# At this point, the following CMake variables should be defined:
#   - NOA_FFTW_INC
#   - NOA_FFTW_DOUBLE_LIB
#   - NOA_FFTW_FLOAT_LIB
#   - NOA_FFTW_DOUBLE_THREADS_LIB   (can be empty)
#   - NOA_FFTW_FLOAT_THREADS_LIB    (can be empty)

# We can therefore create the imported targets.
add_library(fftw3::float INTERFACE IMPORTED)
set_target_properties(fftw3::float
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_FLOAT_LIB}"
        )

add_library(fftw3::double INTERFACE IMPORTED)
set_target_properties(fftw3::double
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_DOUBLE_LIB}"
        )

if (NOA_FFTW_FLOAT_THREADS_LIB)
    add_library(fftw3::float_threads INTERFACE IMPORTED)
    set_target_properties(fftw3::float_threads
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${NOA_FFTW_FLOAT_THREADS_LIB}"
            )
else ()
    add_library(fftw3::float_threads INTERFACE)
endif ()

if (NOA_FFTW_DOUBLE_THREADS_LIB)
    add_library(fftw3::double_threads INTERFACE IMPORTED)
    set_target_properties(fftw3::double_threads
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${NOA_FFTW_DOUBLE_THREADS_LIB}"
            )
else ()
    add_library(fftw3::double_threads INTERFACE)
endif ()

unset(NOA_FFTW_DOUBLE_LIB)
unset(NOA_FFTW_FLOAT_LIB)
unset(NOA_FFTW_DOUBLE_THREADS_LIB)
unset(NOA_FFTW_FLOAT_THREADS_LIB)
unset(NOA_FFTW_INC)
