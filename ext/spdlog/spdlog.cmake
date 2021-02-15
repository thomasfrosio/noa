message(STATUS "Fetching static dependency: spdlog")
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.8.1
)
FetchContent_MakeAvailable(spdlog)

# With CUDA C++17, everything below v1.8.1 doesn't compile when compilation is steered by nvcc. Seems to be related
# to https://github.com/gabime/spdlog/issues/1680. Anyway, with versions > v1.8.1 it compiles.
