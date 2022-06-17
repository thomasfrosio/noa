message(STATUS "spdlog: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

set(spdlog_REPOSITORY https://github.com/gabime/spdlog)
set(spdlog_TAG v1.8.1)

message(STATUS "Repository: ${spdlog_REPOSITORY}")
message(STATUS "Git tag: ${spdlog_TAG}")

include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY ${spdlog_REPOSITORY}

        # With CUDA C++17, everything below v1.8.1 doesn't compile when compilation is steered by nvcc.
        # Seems to be related to https://github.com/gabime/spdlog/issues/1680.
        # Anyway, with versions v1.8.1 it compiles.
        GIT_TAG ${spdlog_TAG}
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})

    # clang++-10 gives an error in format.h line 3677
    # It looks like its because with nvcc FMT_USE_UDL_TEMPLATE is set to 1.
    target_compile_definitions(spdlog PUBLIC FMT_USE_UDL_TEMPLATE=0)
endif()

message(STATUS "New imported target available: spdlog::spdlog")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "spdlog: fetching static dependency... done")
