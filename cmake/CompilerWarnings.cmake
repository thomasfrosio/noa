function(noa_set_project_warnings name)
    option(NOA_WARNINGS_AS_ERRORS "noa - Treat compiler warnings as errors" TRUE)

    # -Wunused # warn on anything being unused
    set(NOA_CLANG_WARNINGS
            -Wall
            -Wextra # reasonable and standard
            -Wshadow # warn the user if a variable declaration shadows one from a parent context
            -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor.
            # This helps catch hard to track down memory errors
            -Wold-style-cast # warn for c-style casts
            -Wcast-align # warn for potential performance problem casts
            -Woverloaded-virtual # warn if you overload (not override) a virtual function
            -Wpedantic # warn if non-standard C++ is used
            -Wconversion # warn on type conversions that may lose data
            -Wsign-conversion # warn on sign conversions
            -Wnull-dereference # warn if a null dereference is detected
            -Wdouble-promotion # warn if float is implicit promoted to double
            -Wformat=2 # warn on security issues around functions that format output (ie printf)
            )

    if (NOA_WARNINGS_AS_ERRORS)
        set(NOA_CLANG_WARNINGS ${NOA_CLANG_WARNINGS} -Werror)
    endif ()

    set(NOA_GCC_WARNINGS
            ${NOA_CLANG_WARNINGS}
            -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
            -Wduplicated-cond # warn if if / else chain has duplicated conditions
            -Wduplicated-branches # warn if if / else branches have duplicated code
            -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
            -Wuseless-cast # warn if you perform a cast to the same type
            )

    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(NOA_WARNINGS ${NOA_CLANG_WARNINGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NOA_WARNINGS ${NOA_GCC_WARNINGS})
    else ()
        message(AUTHOR_WARNING "noa - No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif ()

    target_compile_options(${name} INTERFACE ${NOA_WARNINGS})

endfunction()
