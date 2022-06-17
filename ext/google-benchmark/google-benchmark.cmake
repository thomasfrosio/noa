message(STATUS "google-benchmark: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

set(google-benchmark_REPOSITORY https://github.com/google/benchmark.git)
set(google-benchmark_TAG 713b9177183375c8b1b25595e33daf2a1625df5b)

message(STATUS "Repository: ${google-benchmark_REPOSITORY}")
message(STATUS "Git tag: ${google-benchmark_TAG}")

include(FetchContent)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY ${google-benchmark_REPOSITORY}
        GIT_TAG ${google-benchmark_TAG}
)

FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)

    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    set(BENCHMARK_ENABLE_LTO ${NOA_ENABLE_LTO})
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
endif()

message(STATUS "New imported target available: benchmark::benchmark")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "google-benchmark: fetching static dependency... done")
