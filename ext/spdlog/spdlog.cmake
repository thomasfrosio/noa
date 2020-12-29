message(STATUS "Fetching static dependency: spdlog")
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(spdlog)
