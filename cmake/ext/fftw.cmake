message(STATUS "FFTW3: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET FFTW3::fftw3)
    message(STATUS "Target already exists: FFTW3::fftw3")
else ()
    set(FFTW3_REPOSITORY https://github.com/thomasfrosio/fftw3)
    set(FFTW3_TAG main)

    message(STATUS "Repository: ${FFTW3_REPOSITORY}")
    message(STATUS "Git tag: ${FFTW3_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        FFTW3
        GIT_REPOSITORY ${FFTW3_REPOSITORY}
        GIT_TAG ${FFTW3_TAG}
    )

    # Build with optimizations, regardless of our build mode.
    set(CMAKE_BUILD_TYPE_ ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)

    option(FFTW3_BUILD_TESTS "Build tests" OFF)
    option(FFTW3_ENABLE_OPENMP "Use OpenMP for multithreading" ${NOA_CPU_MULTITHREADED_FFTW3})
    FetchContent_MakeAvailable(FFTW3)

    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_})

    message(STATUS "New imported target available: FFTW3::fftw3, FFTW3::fftw3f")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "FFTW3: fetching static dependency... done")
