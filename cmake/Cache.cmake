if (NOT NOA_ENABLE_CACHE)
    return()
endif ()

set(NOA_CACHE_OPTION "ccache" CACHE STRING "Compiler cache to be used")
set(NOA_CACHE_OPTION_VALUES "ccache" "sccache")
set_property(CACHE NOA_CACHE_OPTION PROPERTY STRINGS ${NOA_CACHE_OPTION_VALUES})
list(
        FIND
        NOA_CACHE_OPTION_VALUES
        ${NOA_CACHE_OPTION}
        NOA_CACHE_OPTION_INDEX)

if (${NOA_CACHE_OPTION_INDEX} EQUAL -1)
    message(
            STATUS
            "Using custom compiler cache system: '${NOA_CACHE_OPTION}', explicitly supported entries are ${NOA_CACHE_OPTION_VALUES}"
    )
endif ()

find_program(NOA_CACHE_BINARY ${NOA_CACHE_OPTION})
if (NOA_CACHE_BINARY)
    message(STATUS "${NOA_CACHE_OPTION} found and enabled")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${NOA_CACHE_BINARY})
else ()
    message(WARNING "${NOA_CACHE_OPTION} is enabled but was not found. Not using it")
endif ()
