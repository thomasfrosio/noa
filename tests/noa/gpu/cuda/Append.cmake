if (NOT NOA_ENABLE_CUDA)
    return()
endif ()

set(TEST_CUDA_SOURCES
    noa/gpu/cuda/TestCUDADevice.cpp
    noa/gpu/cuda/TestCUDAStream.cpp
    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
