# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        Assets.h
        EntryPoint.cpp
        Helpers.h

        noa/common/TestCommonBorderMode.cpp
        noa/common/TestCommonClampCast.cpp
        noa/common/TestCommonHalf.cpp
        noa/common/TestCommonMatrices.cpp
        noa/common/TestCommonString.cpp
        noa/common/TestCommonTraits.cpp
        noa/common/TestCommonTraitsVectors.cpp
        noa/common/TestCommonVectors.cpp
        noa/common/TestCommonStrides.cpp
        noa/common/TestCommonComplex.cpp
        noa/common/TestCommonOS.cpp
        noa/common/TestCommonIO.cpp
        noa/common/files/TestTextFile.cpp
        noa/common/files/TestImageFile.cpp

        noa/common/geometry/TestCommonGeometryEuler.cpp

        noa/cpu/TestCPUStream.cpp

        noa/cpu/fft/TestCPUTransforms.cpp
        noa/cpu/fft/TestCPUResize.cpp
        noa/cpu/fft/TestCPURemap.cpp
        noa/cpu/fft/TestCPUFilters.cpp

        noa/cpu/math/TestCPUEwise.cpp
        noa/cpu/math/TestCPUReduce.cpp
        noa/cpu/math/TestCPUFind.cpp

        noa/cpu/filter/TestCPUSphere.cpp
        noa/cpu/filter/TestCPURectangle.cpp
        noa/cpu/filter/TestCPUCylinder.cpp
        noa/cpu/filter/TestCPUMedian.cpp
        noa/cpu/filter/TestCPUConvolve.cpp

        noa/cpu/memory/TestCPUIndex.cpp
        noa/cpu/memory/TestCPUInitialize.cpp
        noa/cpu/memory/TestCPUPtrHost.cpp
        noa/cpu/memory/TestCPUResize.cpp
        noa/cpu/memory/TestCPUTranspose.cpp

        noa/cpu/geometry/TestCPURotate.cpp
        noa/cpu/geometry/TestCPUScale.cpp
        noa/cpu/geometry/TestCPUShift.cpp
        noa/cpu/geometry/TestCPUTransform.cpp
        noa/cpu/geometry/TestCPUTransformSymmetry.cpp
        noa/cpu/geometry/TestCPUSymmetry.cpp

        noa/cpu/geometry/fft/TestCPUTransformFFT.cpp
        noa/cpu/geometry/fft/TestCPUShiftFFT.cpp

#        noa/cpu/reconstruct/TestCPUProjections.cpp

        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        noa/gpu/cuda/TestCUDADevice.cpp

        noa/gpu/cuda/memory/TestCUDACopy.cpp
        noa/gpu/cuda/memory/TestCUDAIndex.cpp
        noa/gpu/cuda/memory/TestCUDAInitialize.cpp
        noa/gpu/cuda/memory/TestCUDAPtrPinned.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevice.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevicePadded.cpp
        noa/gpu/cuda/memory/TestCUDAPtrArray.cpp
        noa/gpu/cuda/memory/TestCUDAResize.cpp
        noa/gpu/cuda/memory/TestCUDATranspose.cpp

        noa/gpu/cuda/fft/TestCUDATransforms.cpp
        noa/gpu/cuda/fft/TestCUDAResize.cpp
        noa/gpu/cuda/fft/TestCUDARemap.cpp
        noa/gpu/cuda/fft/TestCUDAFilters.cpp

        noa/gpu/cuda/math/TestCUDAEwise.cpp
        noa/gpu/cuda/math/TestCUDAFind.cpp
        noa/gpu/cuda/math/TestCUDAReduce.cpp

        noa/gpu/cuda/filter/TestCUDASphere.cpp
        noa/gpu/cuda/filter/TestCUDACylinder.cpp
        noa/gpu/cuda/filter/TestCUDARectangle.cpp
        noa/gpu/cuda/filter/TestCUDAMedian.cpp
        noa/gpu/cuda/filter/TestCUDAConvolve.cpp

        noa/gpu/cuda/geometry/TestCUDARotate.cpp
        noa/gpu/cuda/geometry/TestCUDAScale.cpp
        noa/gpu/cuda/geometry/TestCUDAShift.cpp
        noa/gpu/cuda/geometry/TestCUDATransform.cpp
        noa/gpu/cuda/geometry/TestCUDATransformSymmetry.cpp
        noa/gpu/cuda/geometry/TestCUDASymmetry.cpp

        noa/gpu/cuda/geometry/fft/TestCUDAApplyFFT.cpp
        noa/gpu/cuda/geometry/fft/TestCUDAShiftFFT.cpp
#
#        noa/gpu/cuda/reconstruct/TestCUDAProjections.cpp

        )

if (NOA_ENABLE_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_tests.cmake)
