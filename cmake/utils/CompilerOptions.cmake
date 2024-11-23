function(noa_set_compiler_options public_target private_target)
    target_compile_features(${public_target} INTERFACE cxx_std_20) # requires at least 20

    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if ("CUDA" IN_LIST languages AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # Make constexpr functions __device__ functions.
        # We use this all over the place in public headers so enforce this on the user...
        target_compile_options(${public_target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)

        # Allow __device__ lambdas. We use this in a few tests, so keep it private.
        target_compile_options(${private_target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

        if (NOT CMAKE_CUDA_HOST_COMPILER) # defaults to the host compiler being the C++ compiler
            target_compile_options(${private_target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: -ccbin ${CMAKE_CXX_COMPILER}>)
        endif ()
    endif ()
endfunction()

function(noa_set_compiler_warnings target enable_warnings_as_errors)
    set(_clang_warnings
        -Wall
        -Wextra                 # reasonable and standard
        -Wunused                # warn on anything being unused
        -Wshadow                # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor      # warn the user if a class with virtual functions has a non-virtual destructor.
        -Wcast-align            # warn for potential performance problem casts
        -Woverloaded-virtual    # warn if you overload (not override) a virtual function
        -Wpedantic              # warn if non-standard C++ is used
        -Wconversion            # warn on type conversions that may lose data
        -Wsign-conversion       # warn on sign conversions
        -Wnull-dereference      # warn if a null dereference is detected
        -Wdouble-promotion      # warn if float is implicit promoted to double
        -Wformat=2              # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough  # warn on statements that fallthrough without an explicit annotation
        # -Wold-style-cast        # warn for c-style casts
    )

    if (${${enable_warnings_as_errors}})
        set(_clang_warnings ${_clang_warnings} -Werror)
    endif ()

    set(_gcc_warnings
        ${_clang_warnings}
        -Wmisleading-indentation    # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond           # warn if if / else chain has duplicated conditions
        -Wduplicated-branches       # warn if if / else branches have duplicated code
        -Wlogical-op                # warn about logical operations being used where bitwise were probably wanted
        -Wuseless-cast              # warn if you perform a cast to the same type
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(_cxx_warnings ${_clang_warnings})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_cxx_warnings ${_gcc_warnings})
    else ()
        set(_cxx_warnings)
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif ()
    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CXX>: ${_cxx_warnings}>)

    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if ("CUDA" IN_LIST languages)
        set(_nvcc_warnings
            --Wreorder                  # Warn when member initializers are reordered
            --Wdefault-stream-launch    # Warn when the stream argument is not provided in the kernel launch syntax
            --Wext-lambda-captures-this # Warn when an extended lambda implicitly captures 'this'
            --forward-unknown-to-host-compiler ${_cxx_warnings} # Forward C++ warnings to host compiler
            )
        # Don't forward --Wpedantic since nvcc generates compiler specific code.
        # -Werror doesn't have the same syntax in nvcc.
        list(REMOVE_ITEM _nvcc_warnings -Wpedantic -Werror)

        if (${${enable_warnings_as_errors}})
            set(_nvcc_warnings ${_nvcc_warnings} -Werror all-warnings)
        endif ()

        if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            set(_cuda_warnings ${_nvcc_warnings})
        elseif ()
            message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CUDA_COMPILER_ID}' compiler.")
        endif ()
        target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: ${_cuda_warnings}>)
    endif ()
endfunction()
