list(APPEND TEST_SOURCES
    noa/runtime/core/TestRuntimeCoreAccessor.cpp
    noa/runtime/core/TestRuntimeCoreIndexing.cpp
    noa/runtime/core/TestRuntimeCoreInterfaces.cpp
    noa/runtime/core/TestRuntimeCoreSpan.cpp

    noa/runtime/cpu/TestRuntimeCPUAllocate.cpp
    noa/runtime/cpu/TestRuntimeCPUDevice.cpp
    noa/runtime/cpu/TestRuntimeCPUEwise.cpp
    noa/runtime/cpu/TestRuntimeCPUIwise.cpp
    noa/runtime/cpu/TestRuntimeCPUReduceAxesEwise.cpp
    noa/runtime/cpu/TestRuntimeCPUReduceAxesIwise.cpp
    noa/runtime/cpu/TestRuntimeCPUReduceEwise.cpp
    noa/runtime/cpu/TestRuntimeCPUReduceIwise.cpp
    noa/runtime/cpu/TestRuntimeCPUStream.cpp

    noa/runtime/TestRuntimeArray.cpp
    noa/runtime/TestRuntimeBlas.cpp
    noa/runtime/TestRuntimeCast.cpp
    noa/runtime/TestRuntimeComplex.cpp
    noa/runtime/TestRuntimeCopy.cpp
    noa/runtime/TestRuntimeDevice.cpp
    noa/runtime/TestRuntimeEwise.cpp
    noa/runtime/TestRuntimeFactory.cpp
    noa/runtime/TestRuntimeIwise.cpp
    noa/runtime/TestRuntimePermute.cpp
    noa/runtime/TestRuntimeRandom.cpp
    noa/runtime/TestRuntimeReduce.cpp
    noa/runtime/TestRuntimeReduceAxes.cpp
    noa/runtime/TestRuntimeReduceBatch.cpp
    noa/runtime/TestRuntimeReduceEwise.cpp
    noa/runtime/TestRuntimeReduceIwise.cpp
    noa/runtime/TestRuntimeResize.cpp
    noa/runtime/TestRuntimeSort.cpp
    noa/runtime/TestRuntimeStream.cpp
    noa/runtime/TestRuntimeSubregion.cpp
    noa/runtime/TestRuntimeView.cpp
)

if (NOA_ENABLE_CUDA)
    list(APPEND TEST_SOURCES
        noa/runtime/cuda/TestRuntimeCUDAAlloc.cpp
        noa/runtime/cuda/TestRuntimeCUDADevice.cpp
        noa/runtime/cuda/TestRuntimeCUDAEwise.cu
        noa/runtime/cuda/TestRuntimeCUDAIwise.cu
        noa/runtime/cuda/TestRuntimeCUDAReduceAxesEwise.cu
        noa/runtime/cuda/TestRuntimeCUDAReduceAxesIwise.cu
        noa/runtime/cuda/TestRuntimeCUDAReduceEwise.cu
        noa/runtime/cuda/TestRuntimeCUDAReduceIwise.cu
        noa/runtime/cuda/TestRuntimeCUDAStream.cpp
        noa/runtime/cuda/TestRuntimeCUDAVectorization.cpp
    )
endif ()
