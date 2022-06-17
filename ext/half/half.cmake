message(STATUS "half-ieee754: fetching header dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

set(half_REPOSITORY https://github.com/ffyr2w/half-ieee754.git)
set(half_TAG master)

message(STATUS "Repository: ${half_REPOSITORY}")
message(STATUS "Git tag: ${half_TAG}")

include(FetchContent)
FetchContent_Declare(half-ieee754
        GIT_REPOSITORY ${half_REPOSITORY}
        GIT_TAG ${half_TAG}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        )

FetchContent_MakeAvailable(half-ieee754)

message(STATUS "New imported target available: half-ieee754")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "half-ieee754: fetching header dependency... done")
