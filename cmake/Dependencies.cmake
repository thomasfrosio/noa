# ---------------------------------------------------------------------------------------
# Manages the project dependencies
# ---------------------------------------------------------------------------------------
include(FetchContent)

message(STATUS "Starting to get the dependencies...")

# Static libraries... which are often fetched.
include(${PROJECT_SOURCE_DIR}/ext/spdlog/spdlog.cmake)

# Shared libraries... which are found.
# include(${PROJECT_SOURCE_DIR}/ext/tiff/tiff.cmake)


message(STATUS "Finishing to get the dependencies...\n")
