###
#
#   1) Find LAPACK
#   2) Then, this module checks that the LAPACKE libraries contains the LAPACKE API.
#      If it doesn't, find lapacke lib
#   3) Find lapacke.h
#
# 2) Check lapacke.h and LAPAC
#  LAPACKE depends on the following libraries:
#   - LAPACK

# LAPACKE depends on LAPACK.
if (NOT LAPACK_FOUND)
    if (LAPACKE_FIND_REQUIRED)
        find_package(LAPACK REQUIRED)
    else ()
        find_package(LAPACK)
    endif ()
elseif (NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    if (LAPACK_LIBRARIES)
        set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
            )
    endif ()
    if (LAPACK_LINKER_FLAGS)
        set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_OPTIONS "${LAPACK_LINKER_FLAGS}"
            )
    endif ()
    if (LAPACK_INCLUDE_DIRS)
        target_include_directories(LAPACK::LAPACK PUBLIC ${LAPACK_INCLUDE_DIRS})
    endif ()
endif ()

# Find header.
if (LAPACKE_FIND_REQUIRED)
    find_file(LAPACKE_INCLUDES REQUIRED NAMES lapacke.h)
else ()
    find_file(LAPACKE_INCLUDES NAMES lapacke.h)
endif ()

get_target_property(LAPACK_LIBRARIES LAPACK::LAPACK INTERFACE_LINK_LIBRARIES)
if (NOT LAPACKE_FIND_QUIETLY)
    if (LAPACKE_INCLUDES)
        message(STATUS "Found LAPACK libraries: ${LAPACK_LIBRARIES}")
        message(STATUS "Found LAPACKE header: ${LAPACKE_INCLUDES}")
    endif()
endif()

set(_lapacke_found_valid)
mark_as_advanced(_lapacke_found_valid)

if (LAPACK_FOUND AND LAPACKE_INCLUDES)
    include(CheckSymbolExists)

    # Find LAPACKE API:
    get_target_property(_lapack_link_options LAPACK::LAPACK INTERFACE_LINK_OPTIONS)
    mark_as_advanced(_lapack_link_options)

    set(CMAKE_REQUIRED_LIBRARIES "${LAPACK_LIBRARIES}")
    if (_lapack_link_options)
        set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${_lapack_link_options}")
    endif ()

    # unset(CBLAS_WORKS CACHE)
    check_symbol_exists(LAPACKE_dgeqrf ${LAPACKE_INCLUDES} LAPACKE_WORKS)
    mark_as_advanced(LAPACKE_WORKS)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if (LAPACKE_WORKS)
        if (NOT LAPACKE_FIND_QUIETLY)
            message(STATUS "Looking for LAPACKE: LAPACK supports the LAPACKE API")
        endif ()
        set(_lapacke_found_valid TRUE)

    else()
        if (NOT LAPACKE_FIND_QUIETLY)
            message(STATUS "Looking for LAPACKE: LAPACK does not supports the LAPACKE API")
        endif ()

        # Find LAPACKE library:
        # This is not supper robust since we only search for the name "*lapacke*"
        # Also, we assume it contains the LAPACKE API it is literally named lapacke.
        set(_old_cmake_lib_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
        if (LAPACKE_STATIC)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
        else ()
            set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif ()

        find_library(LAPACKE_LIBRARIES NAMES lapacke)
        if (LAPACKE_LIBRARIES)
            set(_lapacke_found_valid TRUE)
            if (NOT LAPACKE_FOUND_QUIETLY)
                message(STATUS "Found LAPACKE library: ${LAPACKE_LIBRARIES}")
            endif ()
        endif()

        set(CMAKE_FIND_LIBRARY_SUFFIXES ${_old_cmake_lib_suffix})
    endif ()

    # Create target. We always inherit from LAPACK.
    get_filename_component(LAPACKE_INCLUDE_DIRS ${LAPACKE_INCLUDES} DIRECTORY)
    add_library(LAPACKE::LAPACKE INTERFACE IMPORTED)
    target_link_libraries(LAPACKE::LAPACKE INTERFACE LAPACK::LAPACK)
    target_include_directories(LAPACKE::LAPACKE INTERFACE ${LAPACKE_INCLUDE_DIRS})
    if (LAPACKE_LIBRARIES)
        target_link_libraries(LAPACKE::LAPACKE INTERFACE ${LAPACKE_LIBRARIES})
    endif ()

    if (NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "New IMPORTED target created: LAPACKE::LAPACKE")
    endif()

elseif (LAPACK_FOUND AND NOT LAPACKE_INCLUDES)
    if (NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "lapacke.h could not be found. Please guide the search using LAPACK_ROOT, LAPACKE_ROOT or CMAKE_PREFIX_PATH")
    endif ()
elseif ()
    if (NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "LAPACKE requires LAPACK but LAPACK could not been found. Please guide the search using LAPACK_ROOT, LAPACKE_ROOT or CMAKE_PREFIX_PATH")
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE DEFAULT_MSG
    LAPACKE_INCLUDES
    _lapacke_found_valid)
