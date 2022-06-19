# Find the OpenBLAS library.
# This should work both on Linux and Windows.
#
# The following variables will be used:
#   NOA_OPENBLAS_STATIC:                If true, only static libraries are found, otherwise both static and shared.
#   (ENV) NOA_ENV_OPENBLAS_LIBRARIES:   If set and not empty, the libraries are exclusively searched under this path.
#   (ENV) NOA_ENV_OPENBLAS_INCLUDE:     If set and not empty, the headers (i.e. cblas.h) are exclusively searched under this path.
#
# The following target is created:
#   openblas::openblas
#
# The following variable is set:
#   NOA_OPENBLAS_FOUND

# Log input variables:
message(STATUS "[input] NOA_OPENBLAS_STATIC: ${NOA_OPENBLAS_STATIC}")
message(STATUS "[input (env)] NOA_ENV_OPENBLAS_LIBRARIES: $ENV{NOA_ENV_OPENBLAS_LIBRARIES}")
message(STATUS "[input (env)] NOA_ENV_OPENBLAS_INCLUDE: $ENV{NOA_ENV_OPENBLAS_INCLUDE}")

# Whether to search for static or dynamic libraries.
set(NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (NOA_OPENBLAS_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

if (DEFINED ENV{NOA_ENV_OPENBLAS_LIBRARIES} AND NOT $ENV{NOA_ENV_OPENBLAS_LIBRARIES} STREQUAL "")
    find_library(
            NOA_OPENBLAS_LIB
            NAMES openblas libopenblas
            PATHS $ENV{NOA_ENV_OPENBLAS_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
            REQUIRED
    )
else ()
    find_library(
            NOA_OPENBLAS_LIB
            NAMES openblas libopenblas
            PATHS ${LIB_INSTALL_DIR}
            REQUIRED
    )
endif ()

if (DEFINED ENV{NOA_ENV_OPENBLAS_INCLUDE} AND NOT $ENV{NOA_ENV_OPENBLAS_INCLUDE} STREQUAL "")
    find_path(NOA_OPENBLAS_BLAS_INC
            NAMES "cblas.h"
            PATHS $ENV{NOA_ENV_OPENBLAS_INCLUDE}
            PATH_SUFFIXES "include" "include/openblas"
            NO_DEFAULT_PATH
            REQUIRED
            )
    find_path(NOA_OPENBLAS_LAPACK_INC
            NAMES "lapack.h"
            PATHS $ENV{NOA_ENV_OPENBLAS_INCLUDE}
            PATH_SUFFIXES "include" "include/openblas"
            NO_DEFAULT_PATH
            REQUIRED
            )
else ()
    find_path(NOA_OPENBLAS_BLAS_INC
            NAMES "cblas.h"
            PATHS ${INCLUDE_INSTALL_DIR}
            REQUIRED
            )
    find_path(NOA_OPENBLAS_LAPACK_INC
            NAMES "lapack.h"
            PATHS ${INCLUDE_INSTALL_DIR}
            REQUIRED
            )
endif ()
# Reset to whatever it was:
set(CMAKE_FIND_LIBRARY_SUFFIXES ${NOA_CMAKE_FIND_LIBRARY_SUFFIXES_OLD})

set(NOA_OPENBLAS_INC "${NOA_OPENBLAS_BAS_INC}" "${NOA_OPENBLAS_LAPACK_INC}")
set(NOA_OPENBLAS_LIB_FOUND TRUE)

# Logging:
message(STATUS "[output] NOA_OPENBLAS_LIB_FOUND: ${NOA_OPENBLAS_LIB_FOUND}")
message(STATUS "[output] NOA_OPENBLAS_LIB: ${NOA_OPENBLAS_LIB}")
message(STATUS "[output] NOA_OPENBLAS_BLAS_INC: ${NOA_OPENBLAS_BLAS_INC}")
message(STATUS "[output] NOA_OPENBLAS_LAPACK_INC: ${NOA_OPENBLAS_LAPACK_INC}")

# Targets:
add_library(openblas::openblas INTERFACE IMPORTED)
set_target_properties(openblas::openblas
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NOA_OPENBLAS_INC}"
        INTERFACE_LINK_LIBRARIES "${NOA_OPENBLAS_LIB}"
        )
message(STATUS "New imported target available: openblas::openblas")

mark_as_advanced(
        NOA_OPENBLAS_LIB
        NOA_OPENBLAS_BAS_INC
        NOA_OPENBLAS_LAPACK_INC
)
