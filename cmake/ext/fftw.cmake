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

    set(BUILD_SHARED_LIBS_ ${BUILD_SHARED_LIBS})
    if (NOA_CPU_FFTW3_STATIC)
        set(BUILD_SHARED_LIBS OFF)
    else ()
        set(BUILD_SHARED_LIBS ON)
    endif ()

    set(FFTW3_BUILD_TESTS OFF)
    set(FFTW3_ENABLE_OPENMP ${NOA_CPU_FFTW3_MULTITHREADED})
    FetchContent_MakeAvailable(FFTW3)

    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_})
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_})

    message(STATUS "New imported target available: FFTW3::fftw3, FFTW3::fftw3f")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "FFTW3: fetching static dependency... done")

