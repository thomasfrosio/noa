option(NOA_ENABLE_CACHE "noa - Enable cache if available" ON)
if (NOT NOA_ENABLE_CACHE)
    return()
endif ()

set(NOA_CACHE_OPTION
        "ccache"
        CACHE STRING "noa - Compiler cache to be used")
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
            "noa - Using custom compiler cache system: '${CACHE_OPTION}', explicitly supported entries are ${CACHE_OPTION_VALUES}")
endif ()

find_program(CACHE_BINARY ${NOA_CACHE_OPTION})
if (CACHE_BINARY)
    message(STATUS "noa - ${NOA_CACHE_OPTION} found and enabled")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CACHE_BINARY})
else ()
    message(WARNING "noa - ${NOA_CACHE_OPTION} is enabled but was not found. Not using it")
endif ()
