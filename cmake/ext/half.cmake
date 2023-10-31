message(STATUS "half: fetching header dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET half::half)
    message(STATUS "Target already exists: half::half")
else ()
    set(half_REPOSITORY https://github.com/thomasfrosio/half-ieee754.git)
    set(half_TAG master)

    message(STATUS "Repository: ${half_REPOSITORY}")
    message(STATUS "Git tag: ${half_TAG}")

    include(FetchContent)
    FetchContent_Declare(half
        GIT_REPOSITORY ${half_REPOSITORY}
        GIT_TAG ${half_TAG}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        )
    FetchContent_MakeAvailable(half)

    message(STATUS "New imported target available: half::half")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "half: fetching header dependency... done")
