# Find the FFTW library (only single and double precision versions).
#
# The following variables will be set:
#   NOA_FFTW_LIB            Full paths to all found fftw libraries.
#   NOA_FFTW_INC            fftw include directory paths.
#   NOA_FFTW_DOUBLE_LIB     Full path to fftw3.
#   NOA_FFTW_FLOAT_LIB      Full path to fftw3f.
#
# The following variables will be used:
#   NOA_FFTW_USE_STATIC_LIBS:   If true, only static libraries are found, otherwise both static and shared.
#   (ENV) NOA_FFTW_LIBRARIES:   If set, the libraries are exclusively searched under this path.
#   (ENV) NOA_FFTW_INCLUDE:     If set, the headers (i.e. fftw3.h) are exclusively searched under this path.
#

# Whether to search for static or dynamic libraries.
set(NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (NOA_FFTW_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

# Find
if (DEFINED $ENV{NOA_FFTW_LIBRARIES})
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

if (DEFINED $ENV{NOA_FFTW_INCLUDE})
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
set( CMAKE_FIND_LIBRARY_SUFFIXES ${NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD} )

if (NOT NOA_FFTW_FLOAT_LIB)
    message(FATAL_ERROR "Could not find fftw3f on the system.")
elseif (NOT NOA_FFTW_DOUBLE_LIB)
    message(FATAL_ERROR "Could not find fftw3 on the system.")
elseif (NOT NOA_FFTW_FLOAT_THREADS_LIB)
    message(FATAL_ERROR "Could not find fftw3f threads on the system.")
elseif (NOT NOA_FFTW_DOUBLE_THREADS_LIB)
    message(FATAL_ERROR "Could not find fftw3 threads on the system.")
elseif (NOT NOA_FFTW_INC)
    message(FATAL_ERROR "Could not find the fftw3.h header on the system.")
endif ()

message(STATUS "Found fftw3")
message(STATUS "NOA_FFTW_DOUBLE_LIB: ${NOA_FFTW_DOUBLE_LIB}")
message(STATUS "NOA_FFTW_FLOAT_LIB: ${NOA_FFTW_FLOAT_LIB}")
message(STATUS "NOA_FFTW_DOUBLE_THREADS_LIB: ${NOA_FFTW_DOUBLE_THREADS_LIB}")
message(STATUS "NOA_FFTW_FLOAT_THREADS_LIB: ${NOA_FFTW_FLOAT_THREADS_LIB}")
message(STATUS "NOA_FFTW_INC: ${NOA_FFTW_INC}")

# Create the libraries.
set(NOA_FFTW_LIB ${NOA_FFTW_LIB} ${NOA_FFTW_FLOAT_LIB})
add_library(fftw::float INTERFACE IMPORTED)
set_target_properties(fftw::float
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_FLOAT_LIB}"
        )

set(NOA_FFTW_LIB ${NOA_FFTW_LIB} ${NOA_FFTW_FLOAT_THREADS_LIB})
add_library(fftw::float_threads INTERFACE IMPORTED)
set_target_properties(fftw::float_threads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_FLOAT_THREADS_LIB}"
        )

set(NOA_FFTW_LIB ${NOA_FFTW_LIB} ${NOA_FFTW_DOUBLE_LIB})
add_library(fftw::double INTERFACE IMPORTED)
set_target_properties(fftw::double
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_DOUBLE_LIB}"
        )

set(NOA_FFTW_LIB ${NOA_FFTW_LIB} ${NOA_FFTW_DOUBLE_THREADS_LIB})
add_library(fftw::double_threads INTERFACE IMPORTED)
set_target_properties(fftw::double_threads
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NOA_FFTW_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_FFTW_DOUBLE_THREADS_LIB}"
        )

# Hide from GUI
mark_as_advanced(
        NOA_FFTW_DOUBLE_LIB
        NOA_FFTW_FLOAT_LIB
        NOA_FFTW_DOUBLE_THREADS_LIB
        NOA_FFTW_FLOAT_THREADS_LIB
        NOA_FFTW_INC
)
