if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(TEST_CPU_SOURCES
        noa/cpu/fft/TestCPURemap.cpp
        noa/cpu/fft/TestCPUResize.cpp
        noa/cpu/fft/TestCPUTransforms.cpp

        noa/cpu/math/TestCPUBlas.cpp
        noa/cpu/math/TestCPUEwise.cpp
        noa/cpu/math/TestCPUFind.cpp
        noa/cpu/math/TestCPURandom.cpp
        noa/cpu/math/TestCPUReduce.cpp
        noa/cpu/math/TestCPUSort.cpp

        noa/cpu/memory/TestCPUIndex.cpp
        noa/cpu/memory/TestCPUInitialize.cpp
        noa/cpu/memory/TestCPUPermute.cpp
        noa/cpu/memory/TestCPUPtrHost.cpp
        noa/cpu/memory/TestCPUResize.cpp

        noa/cpu/signal/fft/TestCPUBandpass.cpp
        noa/cpu/signal/fft/TestCPUShiftFFT.cpp
        noa/cpu/signal/fft/TestCPUStandardize.cpp
        noa/cpu/signal/TestCPUConvolve.cpp
        noa/cpu/signal/TestCPUMedian.cpp

        noa/cpu/TestCPUDevice.cpp
        noa/cpu/TestCPUStream.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CPU_SOURCES})
