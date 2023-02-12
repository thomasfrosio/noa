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

# nvcc warns about unrecognized GCC pragma and it seems related to https://github.com/fmtlib/fmt/pull/3057
# __NVCOMPILER is not defined by nvcc (probably just nv++) so here this "GCC optimize" pragma in fmt
# is included and ignored (with a warning). To remove the warning to every CUDA file, add this compile
# option to the fmt target so that it is propagated correctly.
# BTW, this is equivalent to -Xcudafe "--diag_suppress=unrecognized_gcc_pragma", but "Xcudafe"
# is not documented, so prefer to use "--diag-suppress" (without underscore).
target_compile_options(fmt PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --diag-suppress 1675>)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "fmt: fetching static dependency... done")
