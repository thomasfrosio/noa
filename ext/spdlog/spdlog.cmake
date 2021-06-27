message(STATUS "spdlog: fetching from github...")
include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog

        # With CUDA C++17, everything below v1.8.1 doesn't compile when compilation is steered by nvcc.
        # Seems to be related to https://github.com/gabime/spdlog/issues/1680.
        # Anyway, with versions v1.8.1 it compiles.
        GIT_TAG v1.8.1
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})

    # clang++-10 gives an error in format.h line 3677
    # It looks like with nvcc FMT_USE_UDL_TEMPLATE is set to 1.
    target_compile_definitions(spdlog PUBLIC FMT_USE_UDL_TEMPLATE=0)
endif()
message("")
