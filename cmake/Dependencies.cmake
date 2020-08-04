include(FetchContent)

# Register, download and add the directories of fmt.
message("######## Load {fmt} ########")
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG 7.0.2
)
FetchContent_MakeAvailable(fmt)

# Register spdlog
message("######## Load spdlog ########")
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.7.0
)
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "Use external fmt for spdlog")
    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif ()
