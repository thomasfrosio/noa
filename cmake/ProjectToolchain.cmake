# Set the build type and print some info about the toolchain.
macro(noa_set_toolchain enable_cuda find_and_set_cuda_architecture)
    if (${enable_cuda})
        include(cmake/utils/CUDA.cmake)
        noa_enable_cuda()
        if (${find_and_set_cuda_architecture})
            noa_cuda_find_and_set_architecture()
        endif ()
    endif ()

    message(STATUS "--------------------------------------")
    message(STATUS "Toolchain...")
    list(APPEND CMAKE_MESSAGE_INDENT "   ")

    # Use CMAKE_CONFIGURATION_TYPES or CMAKE_BUILD_TYPE
    get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (isMultiConfig)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo" CACHE STRING "")
        message(STATUS "CMAKE_CONFIGURATION_TYPES (multi-config): ${CMAKE_CONFIGURATION_TYPES}")
    else ()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
        message(STATUS "CMAKE_BUILD_TYPE (single-config): ${CMAKE_BUILD_TYPE}")

        # Set the possible values of build type for cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
    endif ()

    message(STATUS "CMAKE_INTERPROCEDURAL_OPTIMIZATION: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
    message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
    message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if ("CUDA" IN_LIST languages)
        message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
        if (CMAKE_CUDA_HOST_COMPILER)
            message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
        endif ()
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()


    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(STATUS "Toolchain... done")
    message(STATUS "--------------------------------------\n")

endmacro()
