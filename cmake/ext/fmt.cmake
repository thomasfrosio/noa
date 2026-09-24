message(STATUS "fmt: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET fmt::fmt)
    message(STATUS "Target already exists: fmt::fmt")
else ()
    set(fmt_REPOSITORY https://github.com/fmtlib/fmt)
    set(fmt_TAG 12.2.0) # Sep 7, 2025, v11.2.1, up from v11.1.4 to fix #4477

    message(STATUS "Repository: ${fmt_REPOSITORY}")
    message(STATUS "Git tag: ${fmt_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY ${fmt_REPOSITORY}
        GIT_TAG ${fmt_TAG}
    )
    # we require fmt's exported targets and package config.
    # This is fixed in https://github.com/fmtlib/fmt/pull/4850
    set(FMT_INSTALL ON)
    FetchContent_MakeAvailable(fmt)

    message(STATUS "New imported target available: fmt::fmt")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "fmt: fetching static dependency... done")
