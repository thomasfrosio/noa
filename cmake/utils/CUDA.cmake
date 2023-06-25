# Find the local CUDA architecture for all current devices.
macro(noa_cuda_find_and_set_architecture)
    set(IFILE ${PROJECT_SOURCE_DIR}/cmake/utils/CUDASelectArch.cu)
    set(OFILE ${CMAKE_CURRENT_BINARY_DIR}/CUDASelectArch.o)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} ${IFILE} -o ${OFILE})
    execute_process(COMMAND ${OFILE}
        RESULT_VARIABLE noa_cuda_architecture_has_failed # Returns -1 if failed, 0 otherwise
        OUTPUT_VARIABLE noa_cuda_architectures)

    if (NOT ${noa_cuda_architecture_has_failed} EQUAL 0)
        message(FATAL_ERROR "Could not detect the CUDA architecture on this system. CUDA_ARCHITECTURES would have to be set manually")
    endif ()

    set(CMAKE_CUDA_ARCHITECTURES ${noa_cuda_architectures})
endmacro()

macro(noa_enable_cuda)
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
endmacro()
