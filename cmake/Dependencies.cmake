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

message(STATUS "Finishing to fetch dependencies...\n")
