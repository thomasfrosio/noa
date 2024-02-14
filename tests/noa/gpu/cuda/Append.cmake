if (NOT NOA_ENABLE_CUDA)
    return()
endif ()

set(TEST_CUDA_SOURCES
    noa/gpu/cuda/TestCUDADevice.cpp
    noa/gpu/cuda/TestCUDAStream.cpp
    noa/gpu/cuda/TestCUDAIwise.cu
    noa/gpu/cuda/TestCUDAEwise.cu
    noa/gpu/cuda/TestCUDAReduceEwise.cu
    noa/gpu/cuda/TestCUDAReduceIwise.cu
    noa/gpu/cuda/TestCUDAReduceAxesEwise.cu
    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
