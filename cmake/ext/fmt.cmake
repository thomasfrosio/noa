message(STATUS "fmt: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET fmt::fmt)
    message(STATUS "Target already exists: fmt::fmt")
else ()
    set(fmt_REPOSITORY https://github.com/fmtlib/fmt)
    set(fmt_TAG 9.1.0)

    message(STATUS "Repository: ${fmt_REPOSITORY}")
    message(STATUS "Git tag: ${fmt_TAG}")

    include(FetchContent)
    FetchContent_Declare(
            fmt
            GIT_REPOSITORY ${fmt_REPOSITORY}
            GIT_TAG ${fmt_TAG}
    )
    # See https://github.com/fmtlib/fmt/pull/3264
    option(FMT_INSTALL "Enable installation for the {fmt} project." ON)
    FetchContent_MakeAvailable(fmt)

    message(STATUS "New imported target available: fmt::fmt")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "fmt: fetching static dependency... done")
