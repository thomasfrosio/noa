if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(TEST_CPU_SOURCES
        noa/cpu/fft/TestCPUFilters.cpp
        noa/cpu/fft/TestCPURemap.cpp
        noa/cpu/fft/TestCPUResize.cpp
        noa/cpu/fft/TestCPUTransforms.cpp

        noa/cpu/filter/TestCPUConvolve.cpp
        noa/cpu/filter/TestCPUCylinder.cpp
        noa/cpu/filter/TestCPUMedian.cpp
        noa/cpu/filter/TestCPURectangle.cpp
        noa/cpu/filter/TestCPUSphere.cpp

        noa/cpu/geometry/fft/TestCPUShiftFFT.cpp
        noa/cpu/geometry/fft/TestCPUTransformFFT.cpp
        noa/cpu/geometry/TestCPURotate.cpp
        noa/cpu/geometry/TestCPUScale.cpp
        noa/cpu/geometry/TestCPUShift.cpp
        noa/cpu/geometry/TestCPUSymmetry.cpp
        noa/cpu/geometry/TestCPUTransform.cpp
        noa/cpu/geometry/TestCPUTransformSymmetry.cpp
        # noa/cpu/reconstruct/TestCPUProjections.cpp

        noa/cpu/math/TestCPUEwise.cpp
        noa/cpu/math/TestCPUFind.cpp
        noa/cpu/math/TestCPURandom.cpp
        noa/cpu/math/TestCPUReduce.cpp

        noa/cpu/memory/TestCPUIndex.cpp
        noa/cpu/memory/TestCPUInitialize.cpp
        noa/cpu/memory/TestCPUPtrHost.cpp
        noa/cpu/memory/TestCPUResize.cpp
        noa/cpu/memory/TestCPUTranspose.cpp

        noa/cpu/TestCPUDevice.cpp
        noa/cpu/TestCPUStream.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CPU_SOURCES})
