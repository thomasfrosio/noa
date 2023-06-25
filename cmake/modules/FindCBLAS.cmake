# Find the BLAS libraries and check CBLAS is supported.
# REQUIRE and QUIET are respected, but this module doesn't support versioning.
#
# The following variables are used:
#   BLA_VENDOR  (from FindBLAS.cmake)
#   BLA_ROOT    (from FindBLAS.cmake)
#   BLA_STATIC  (from FindBLAS.cmake)
#
# The following target is created:
#   CBLAS::CBLAS:
#
# Note:  if the GEMM3M complex functions are supported by the library,
#        the compile definition CBLAS_HAS_GEMM3M is added to the target.
#
# The following variables are set:
#   CBLAS_FOUND
#   CBLAS_LIBRARIES
#   CBLAS_INCLUDES
#   CBLAS_GEMM3M_FOUND
#   CBLAS_OpenBLAS_FOUND
#   CBLAS_OpenBLAS_THREAD_MODEL_FOUND
#   CBLAS_OpenBLAS_SERIAL_FOUND
#   CBLAS_OpenBLAS_THREADS_FOUND
#   CBLAS_OpenBLAS_OPENMP_FOUND

# CBLAS depends on BLAS.
if (NOT BLAS_FOUND)
    if (CBLAS_FIND_REQUIRED)
        find_package(BLAS REQUIRED)
    else ()
        find_package(BLAS)
    endif ()
elseif (NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    if (BLAS_LIBRARIES)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
            )
    endif ()
    if (BLAS_LINKER_FLAGS)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_OPTIONS "${BLAS_LINKER_FLAGS}"
            )
    endif ()
    if (BLAS_INCLUDE_DIRS)
        target_include_directories(BLAS::BLAS PUBLIC ${BLAS_INCLUDE_DIRS})
    endif ()
endif ()

# Find header.
if (CBLAS_FIND_REQUIRED)
    find_file(CBLAS_INCLUDES REQUIRED NAMES cblas.h)
else ()
    find_file(CBLAS_INCLUDES NAMES cblas.h)
endif ()

get_target_property(CBLAS_LIBRARIES BLAS::BLAS INTERFACE_LINK_LIBRARIES)
if (NOT CBLAS_FIND_QUIETLY)
    if (CBLAS_INCLUDES)
        message(STATUS "Found BLAS libraries: ${CBLAS_LIBRARIES}")
        message(STATUS "Found CBLAS header: ${CBLAS_INCLUDES}")
    endif()
endif()

if (BLAS_FOUND AND CBLAS_INCLUDES)
    include(CheckSymbolExists)

    # Find CBLAS API:
    get_target_property(_blas_link_options BLAS::BLAS INTERFACE_LINK_OPTIONS)
    mark_as_advanced(_blas_link_options)

    set(CMAKE_REQUIRED_LIBRARIES "${CBLAS_LIBRARIES}")
    if (_blas_link_options)
        set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${_blas_link_options}")
    endif ()

    # unset(CBLAS_WORKS CACHE)
    check_symbol_exists(cblas_dscal ${CBLAS_INCLUDES} CBLAS_WORKS)
    check_symbol_exists(cblas_zgemm3m ${CBLAS_INCLUDES} CBLAS_GEMM3M_FOUND)
    mark_as_advanced(CBLAS_WORKS)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if (CBLAS_WORKS)
        if (NOT CBLAS_FIND_QUIETLY)
            message(STATUS "Looking for CBLAS: BLAS supports the CBLAS API")
        endif ()

        # Create target
        get_filename_component(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDES} DIRECTORY)
        add_library(CBLAS::CBLAS INTERFACE IMPORTED)
        target_link_libraries(CBLAS::CBLAS INTERFACE BLAS::BLAS)
        set_target_properties(CBLAS::CBLAS
            PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            INTERFACE_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_DIRS}
            )
        if (CBLAS_ZGEMM3M_FOUND)
            target_compile_definitions(CBLAS::CBLAS INTERFACE CBLAS_HAS_GEMM3M)
        endif ()

        if (NOT CBLAS_FIND_QUIETLY)
            message(STATUS "New IMPORTED target created: CBLAS::CBLAS")
        endif()

    else()
        if (NOT CBLAS_FIND_QUIETLY)
            message(STATUS "BLAS libraries don't support CBLAS. Please look for another BLAS with CBLAS support")
        endif()
    endif ()

elseif (BLAS_FOUND AND NOT CBLAS_INCLUDES)
    if (NOT CBLAS_FIND_QUIETLY)
        message(STATUS "cblas.h could not be found. Please guide the search using BLAS_ROOT, CBLAS_ROOT or CMAKE_PREFIX_PATH")
    endif ()
elseif ()
    if (NOT CBLAS_FIND_QUIETLY)
        message(STATUS "CBLAS requires BLAS but BLAS could not been found. Please guide the search using BLAS_ROOT, CBLAS_ROOT or CMAKE_PREFIX_PATH")
    endif ()
endif ()

# Handle REQUIRED:
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG
    CBLAS_INCLUDES
    CBLAS_WORKS)

# ---------------------------------------------------------------------------------------
# Optional checks if OpenBLAS
# ---------------------------------------------------------------------------------------
# First check if the library is OpenBLAS.
get_filename_component(_cblas_filename ${CBLAS_LIBRARIES} NAME_WE)
string(FIND ${_cblas_filename} "openblas" _cblas_index)
if (NOT _cblas_index STREQUAL "-1")
    set(BLA_VENDOR OpenBLAS)
    set(CBLAS_OpenBLAS_FOUND TRUE)
endif ()

# Then check if cblas.h is from OpenBLAS.
set(_openblas_is_cblas)
if (BLA_VENDOR STREQUAL "OpenBLAS")
    file(STRINGS "${CBLAS_INCLUDES}" _openblas_is_cblas REGEX "openblas_get_parallel")
endif()

# Then check the threading model.
if (_openblas_is_cblas)
    # Try get the threading model from the runtime API since OpenBLAS doesn't seem
    # to save any information in its headers regarding the threading model.
    set(_openblas_tm_test_dir ${CMAKE_BINARY_DIR}/FindOpenBlas)
    set(_openblas_tm_filename ${_openblas_tm_test_dir}/TestThreadModel.cpp)
    set(_openblas_tm_src [=[
#include <cblas.h>
int main() {
    int th_model = openblas_get_parallel()\;
    switch(th_model) {
        case OPENBLAS_SEQUENTIAL:
            return 0\;
        case OPENBLAS_THREAD:
            return 1\;
        case OPENBLAS_OPENMP:
            return 2\;
    }
    return -1\;
}
]=])

    file(WRITE ${_openblas_tm_filename} ${_openblas_tm_src})
    get_filename_component(_openblas_tm_include_dir ${CBLAS_INCLUDES} DIRECTORY)
    if (NOT DEFINED _openblas_tm_run_result)
        try_run(_openblas_tm_run_result _openblas_tm_compile_result
            ${_openblas_tm_test_dir} ${_openblas_tm_filename}
            LINK_LIBRARIES CBLAS::CBLAS
            )
    endif ()

    set(_openblas_tm_failed OFF)
    if (NOT _openblas_tm_compile_result)
        set(_openblas_tm_failed ON)
    elseif(${_openblas_tm_run_result} STREQUAL "FAILED_TO_RUN")
        set(_openblas_tm_failed ON)
    elseif(${_openblas_tm_run_result} STREQUAL "-1")
        set(_openblas_tm_failed ON)
    endif()

    set(CBLAS_OpenBLAS_THREAD_MODEL_FOUND FALSE)
    set(CBLAS_OpenBLAS_SERIAL_FOUND       FALSE)
    set(CBLAS_OpenBLAS_THREADS_FOUND      FALSE)
    set(CBLAS_OpenBLAS_OPENMP_FOUND       FALSE)
    if (NOT _openblas_tm_failed)
        if (${_openblas_tm_run_result} STREQUAL "0")
            set(CBLAS_OpenBLAS_THREAD_MODEL_FOUND TRUE)
            set(CBLAS_OpenBLAS_SERIAL_FOUND       TRUE)
        elseif(${_openblas_tm_run_result} STREQUAL "1")
            set(CBLAS_OpenBLAS_THREAD_MODEL_FOUND TRUE)
            set(CBLAS_OpenBLAS_THREADS_FOUND      TRUE)
            find_package(Threads)
            target_link_libraries(CBLAS::CBLAS INTERFACE Threads::Threads)
        elseif(${_openblas_tm_run_result} STREQUAL "2")
            set(CBLAS_OpenBLAS_THREAD_MODEL_FOUND TRUE)
            set(CBLAS_OpenBLAS_OPENMP_FOUND       TRUE)
            find_package(OpenMP)
            if (TARGET OpenMP::OpenMP_C)
                target_link_libraries(CBLAS::CBLAS INTERFACE OpenMP::OpenMP_C)
            elseif (TARGET OpenMP::OpenMP_CXX)
                target_link_libraries(CBLAS::CBLAS INTERFACE OpenMP::OpenMP_CXX)
            endif ()
        endif ()
    endif()

    if (NOT CBLAS_FIND_QUIETLY)
        if (NOT CBLAS_OpenBLAS_THREAD_MODEL_FOUND)
            message(STATUS "Could not find OpenBLAS threading model")
        else ()
            message(STATUS "Found OpenBLAS serial: ${CBLAS_OpenBLAS_SERIAL_FOUND}")
            message(STATUS "Found OpenBLAS threads: ${CBLAS_OpenBLAS_THREADS_FOUND}")
            message(STATUS "Found OpenBLAS OpenMP: ${CBLAS_OpenBLAS_OPENMP_FOUND}")
        endif ()
    endif()

    mark_as_advanced(
        _openblas_is_cblas
        _openblas_tm_test_dir
        _openblas_tm_filename
        _openblas_tm_src
        _openblas_tm_include_dir
        _openblas_tm_compile_result
        _openblas_tm_run_result
        _openblas_tm_failed
    )
endif()
