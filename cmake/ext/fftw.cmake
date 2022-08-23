message(STATUS "FFTW3: searching for existing libraries...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET FFTW3::FFTW3_float  AND
    TARGET FFTW3::FFTW3_double AND
    (NOT FFTW3_OPENMP OR  (TARGET FFTW3::FFTW3_float_openmp  AND FFTW3::FFTW3_double_openmp)) AND
    (NOT FFTW3_THREADS OR (TARGET FFTW3::FFTW3_float_threads AND FFTW3::FFTW3_double_threads)))

    message(STATUS "Target already exists: FFTW3::FFTW3_float")
    message(STATUS "Target already exists: FFTW3::FFTW3_double")
    if (TARGET FFTW3::FFTW3_threads)
        message(STATUS "Target already exists: FFTW3::FFTW3_float_threads")
        message(STATUS "Target already exists: FFTW3::FFTW3_double_threads")
    endif ()
    if (TARGET FFTW3::FFTW3_openmp)
        message(STATUS "Target already exists: FFTW3::FFTW3_float_openmp")
        message(STATUS "Target already exists: FFTW3::FFTW3_double_openmp")
    endif ()
else ()
    message(STATUS "[in] FFTW3_THREADS: ${FFTW3_THREADS}")
    message(STATUS "[in] FFTW3_OPENMP: ${FFTW3_OPENMP}")
    message(STATUS "[in] FFTW3_STATIC: ${FFTW3_STATIC}")

    find_package(FFTW3 REQUIRED)
endif ()

if (FFTW3_OPENMP_FOUND)
    set(FFTW3_TARGETS FFTW3::FFTW3_float_openmp FFTW3::FFTW3_double_openmp)
elseif(FFTW3_THREADS_FOUND)
    set(FFTW3_TARGETS FFTW3::FFTW3_float_threads FFTW3::FFTW3_double_threads)
else()
    set(FFTW3_TARGETS FFTW3::FFTW3_float FFTW3::FFTW3_double)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "FFTW3: searching for existing libraries... done")

# Note: FetchContent is not very practical with non-CMake projects.
#       FFTW added CMake support in 3.3.7 but it seems to be experimental even in 3.3.8.
# TODO: Add support for FetchContent or ExternalProject_Add.
