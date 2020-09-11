# ---------------------------------------------------------------------------------------
# Manages the project dependencies
# ---------------------------------------------------------------------------------------
include(FetchContent)

message(STATUS "Starting to fetch the noa dependencies...")

# spdlog
message(STATUS "Dependency: spdlog")
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(spdlog)

#FetchContent_GetProperties(spdlog)
#if (NOT spdlog_POPULATED)
#    FetchContent_Populate(spdlog)
#    set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "Use external fmt for spdlog")
#    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
#endif ()

message(STATUS "Finishing to fetch dependencies...\n")
