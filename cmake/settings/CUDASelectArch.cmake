# Find the local CUDA architecture for all current devices.
# Results are set in NOA_CUDA_ARCH and can be passed to CUDA_ARCHITECTURES.
set(IFILE ${PROJECT_SOURCE_DIR}/cmake/settings/CUDASelectArch.cu)
set(OFILE ${CMAKE_CURRENT_BINARY_DIR}/CUDASelectArch.o)
execute_process(COMMAND ${CMAKE_CUDA_COMPILER} ${IFILE} -o ${OFILE})
execute_process(COMMAND ${OFILE}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE NOA_CUDA_ARCH)

if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
    message(FATAL_ERROR "Could not detect the CUDA architecture on this system. NOA_CUDA_ARCH would have to be set manually")
endif ()
