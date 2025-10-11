message(STATUS "google-benchmark: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET benchmark::benchmark)
    message(STATUS "Target already exists: benchmark::benchmark")
else ()
    set(google-benchmark_REPOSITORY https://github.com/google/benchmark.git)
    set(google-benchmark_TAG 713b9177183375c8b1b25595e33daf2a1625df5b)

    message(STATUS "Repository: ${google-benchmark_REPOSITORY}")
    message(STATUS "Git tag: ${google-benchmark_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY ${google-benchmark_REPOSITORY}
        GIT_TAG ${google-benchmark_TAG}
        EXCLUDE_FROM_ALL # no install
    )

    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    set(BENCHMARK_ENABLE_LTO OFF)
    FetchContent_MakeAvailable(benchmark)

    message(STATUS "New imported target available: benchmark::benchmark")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "google-benchmark: fetching static dependency... done")
