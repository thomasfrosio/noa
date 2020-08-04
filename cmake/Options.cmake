# Set and unset the options.

function(setOption)
    option(NOA_ENABLE_CACHE "Enable cache if available" OFF)
    option(NOA_ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
    option(NOA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
endfunction()

function(unsetOption)
    unset(NOA_ENABLE_CACHE CACHE)
    unset(NOA_ENABLE_IPO CACHE)
    unset(NOA_WARNINGS_AS_ERRORS CACHE)
endfunction()
