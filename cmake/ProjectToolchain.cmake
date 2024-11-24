# Set the build type and print some info about the toolchain.
macro(noa_set_toolchain enable_cuda)
    if (${enable_cuda})
        # If not specified, use native. CMake 3.24 supports passing "native".
        # This needs to be done before enabling CUDA.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "native")
        endif ()

        include(CheckLanguage)
        check_language(CUDA)
        if (CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            # Use the same C++ standard if not specified.
            # This is not used by noa targets directly, but can be useful for dependencies.
            if ("${CMAKE_CUDA_STANDARD}" STREQUAL "")
                set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
            endif ()
        else ()
            message(FATAL_ERROR "CUDA is required, but no CUDA support is detected")
        endif ()
    endif ()

    message(STATUS "--------------------------------------")
    message(STATUS "Toolchain...")
    list(APPEND CMAKE_MESSAGE_INDENT "   ")

    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
    message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if ("CUDA" IN_LIST languages)
        message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
        if (CMAKE_CUDA_HOST_COMPILER)
            message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
            if (NOT ${CMAKE_CUDA_HOST_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
                message(WARNING "CUDA host compiler is not equal to the C++ compiler, which is likely to cause some issues!")
            endif ()
        endif ()
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(STATUS "Toolchain... done")
    message(STATUS "--------------------------------------\n")

endmacro()
