function(noa_get_cxx_compiler_warnings enable_warnings_as_errors)
    set(_clang_warnings
        -Wall
        -Wextra # reasonable and standard
        -Wunused # warn on anything being unused
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor.
        # This helps catch hard to track down memory errors
        # -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
    )

    if (enable_warnings_as_errors)
        set(_clang_warnings ${_clang_warnings} -Werror)
    endif ()

    set(_gcc_warnings
        ${_clang_warnings}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(NOA_CXX_COMPILER_WARNINGS ${_clang_warnings} PARENT_SCOPE)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NOA_CXX_COMPILER_WARNINGS ${_gcc_warnings} PARENT_SCOPE)
    else ()
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif ()
endfunction()

function(noa_get_cuda_compiler_warnings enable_warnings_as_errors)
    if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CUDA_COMPILER_ID}' compiler.")
    endif ()

    # We only support nvcc atm, so forward all C++ warnings in CUDA files
    # except --Wpedantic (since nvcc generates compiler specific code).
    noa_get_cxx_compiler_warnings(${enable_warnings_as_errors})
    set(_cuda_warnings
        --forward-unknown-to-host-compiler
        --Wreorder # Generate warnings when member initializers are reordered.
        --Wdefault-stream-launch # Generate warning when an explicit stream argument is not provided in the <<<...>>> kernel launch syntax.
        --Wext-lambda-captures-this # Generate warning when an extended lambda implicitly captures 'this'.
        ${NOA_CXX_COMPILER_WARNINGS})
    list(REMOVE_ITEM _cuda_warnings "-Wpedantic" "-Werror")

    if (${enable_warnings_as_errors})
        set(_cuda_warnings ${_cuda_warnings} -Werror all-warnings)
    endif ()
    set(NOA_CUDA_COMPILER_WARNINGS ${_cuda_warnings} PARENT_SCOPE)
endfunction()

function(noa_set_cxx_compiler_warnings target enable_warnings_as_errors)
    noa_get_cxx_compiler_warnings(${${enable_warnings_as_errors}})
    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CXX>: ${NOA_CXX_COMPILER_WARNINGS}>)
endfunction()

function(noa_set_cuda_compiler_warnings target enable_warnings_as_errors)
    noa_get_cuda_compiler_warnings(${${enable_warnings_as_errors}})
    target_compile_options(${target} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: ${NOA_CUDA_COMPILER_WARNINGS}>)
endfunction()

function(noa_set_compiler_options target)
    target_compile_features(prj_common_option INTERFACE cxx_std_20) # All of our targets are C++20

    set(_cuda_options --expt-relaxed-constexpr --extended-lambda)
    target_compile_options(prj_common_option INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: ${_cuda_options}>)
endfunction()
