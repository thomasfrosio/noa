message(STATUS "Catch2: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET Catch2::Catch2)
    message(STATUS "Target already exists: Catch2::Catch2")
else ()
    set(Catch2_REPOSITORY https://github.com/catchorg/Catch2.git)
    set(Catch2_TAG v3.8.0)

    message(STATUS "Repository: ${Catch2_REPOSITORY}")
    message(STATUS "Git tag: ${Catch2_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY ${Catch2_REPOSITORY}
        GIT_TAG ${Catch2_TAG}
    )
    FetchContent_MakeAvailable(Catch2)

    message(STATUS "New imported target available: Catch2::Catch2")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "Catch2: fetching static dependency... done")
