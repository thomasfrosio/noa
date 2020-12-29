message(STATUS "Fetching static dependency: Catch2")
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.3
)
FetchContent_MakeAvailable(Catch2)
