# ---------------------------------------------------------------------------------------
# Manages the project dependencies
# ---------------------------------------------------------------------------------------
include(FetchContent)

message(STATUS "Starting to fetch the dependencies...")

# spdlog
include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)

message(STATUS "Finishing to fetch the dependencies...\n")
