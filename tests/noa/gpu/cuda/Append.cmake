if (NOT NOA_ENABLE_CUDA)
    return()
endif()

set(TEST_CUDA_SOURCES
        # noa/gpu/cuda/fft/TestCUDAFilters.cpp
        # noa/gpu/cuda/fft/TestCUDARemap.cpp
        # noa/gpu/cuda/fft/TestCUDAResize.cpp
        # noa/gpu/cuda/fft/TestCUDATransforms.cpp
        # noa/gpu/cuda/reconstruct/TestCUDAProjections.cpp

        noa/gpu/cuda/geometry/fft/TestCUDAShiftFFT.cpp
        noa/gpu/cuda/geometry/fft/TestCUDATransformFFT.cpp
        noa/gpu/cuda/geometry/TestCUDARotate.cpp
        noa/gpu/cuda/geometry/TestCUDAScale.cpp
        noa/gpu/cuda/geometry/TestCUDAShift.cpp
        noa/gpu/cuda/geometry/TestCUDASymmetry.cpp
        noa/gpu/cuda/geometry/TestCUDATransform.cpp
        noa/gpu/cuda/geometry/TestCUDATransformSymmetry.cpp

        noa/gpu/cuda/math/TestCUDAEwise.cpp
        noa/gpu/cuda/math/TestCUDAFind.cpp
        noa/gpu/cuda/math/TestCUDAReduce.cpp

        noa/gpu/cuda/filter/TestCUDAConvolve.cpp
        noa/gpu/cuda/filter/TestCUDACylinder.cpp
        noa/gpu/cuda/filter/TestCUDAMedian.cpp
        noa/gpu/cuda/filter/TestCUDARectangle.cpp
        noa/gpu/cuda/filter/TestCUDASphere.cpp

        noa/gpu/cuda/memory/TestCUDACopy.cpp
        noa/gpu/cuda/memory/TestCUDAIndex.cpp
        noa/gpu/cuda/memory/TestCUDAInitialize.cpp
        noa/gpu/cuda/memory/TestCUDAPtrArray.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevice.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevicePadded.cpp
        noa/gpu/cuda/memory/TestCUDAPtrPinned.cpp
        noa/gpu/cuda/memory/TestCUDAResize.cpp
        noa/gpu/cuda/memory/TestCUDATranspose.cpp

        noa/gpu/cuda/TestCUDADevice.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
