# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        Assets.h
        EntryPoint.cpp
        Helpers.h

        noa/common/TestCommonBorderMode.cpp
        noa/common/TestCommonClampCast.cpp
        noa/common/TestCommonMatrices.cpp
        noa/common/TestCommonString.cpp
        noa/common/TestCommonTraits.cpp
        noa/common/TestCommonTraitsVectors.cpp
        noa/common/TestCommonVectors.cpp
        noa/common/TestCommonComplex.cpp
        noa/common/TestCommonOS.cpp
        noa/common/TestCommonIO.cpp
        noa/common/files/TestTextFile.cpp
        noa/common/files/TestImageFile.cpp

        noa/cpu/fft/TestCPUTransforms.cpp
        noa/cpu/fft/TestCPUResize.cpp
        noa/cpu/fft/TestCPURemap.cpp
        noa/cpu/fft/TestCPUFilters.cpp

        noa/cpu/math/TestCPUArithmetics.cpp
        noa/cpu/math/TestCPUReductions.cpp
        noa/cpu/math/TestCPUArithmeticsComposite.cpp
        noa/cpu/math/TestCPUBooleans.cpp
        noa/cpu/math/TestCPUGenerics.cpp
        noa/cpu/math/TestCPUIndexes.cpp

        noa/cpu/filter/TestCPUSphere.cpp
        noa/cpu/filter/TestCPURectangle.cpp
        noa/cpu/filter/TestCPUCylinder.cpp
        noa/cpu/filter/TestCPUMedian.cpp
        noa/cpu/filter/TestCPUConvolve.cpp

        noa/cpu/memory/TestCPUPtrHost.cpp
        noa/cpu/memory/TestCPUResize.cpp
        noa/cpu/memory/TestCPUIndex.cpp
        noa/cpu/memory/TestCPUTranspose.cpp

        noa/cpu/transform/TestCPURotate.cpp
        noa/cpu/transform/TestCPUScale.cpp
        noa/cpu/transform/TestCPUTranslate.cpp
        noa/cpu/transform/TestCPUApply.cpp
        noa/cpu/transform/TestCPUApplySymmetry.cpp
        noa/cpu/transform/TestCPUSymmetry.cpp

        noa/cpu/transform/fft/TestCPUApplyFFT.cpp
        noa/cpu/transform/fft/TestCPUShiftFFT.cpp


        noa/cpu/reconstruct/TestCPUProjections.cpp

        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        noa/gpu/cuda/util/TestCUDADevice.cpp

        noa/gpu/cuda/memory/TestCUDAPtrPinned.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevice.cpp
        noa/gpu/cuda/memory/TestCUDAPtrDevicePadded.cpp
        noa/gpu/cuda/memory/TestCUDAPtrArray.cpp
        noa/gpu/cuda/memory/TestCUDACopy.cpp
        noa/gpu/cuda/memory/TestCUDAResize.cpp
        noa/gpu/cuda/memory/TestCUDAIndex.cpp
        noa/gpu/cuda/memory/TestCUDATranspose.cpp

        noa/gpu/cuda/fft/TestCUDATransforms.cpp
        noa/gpu/cuda/fft/TestCUDAResize.cpp
        noa/gpu/cuda/fft/TestCUDARemap.cpp
        noa/gpu/cuda/fft/TestCUDAFilters.cpp

        noa/gpu/cuda/math/TestCUDAArithmetics.cpp
        noa/gpu/cuda/math/TestCUDAArithmeticsComposite.cpp
        noa/gpu/cuda/math/TestCUDABooleans.cpp
        noa/gpu/cuda/math/TestCUDAGenerics.cpp
        noa/gpu/cuda/math/TestCUDAIndexes.cpp
        noa/gpu/cuda/math/TestCUDAReductions.cpp

        noa/gpu/cuda/filter/TestCUDASphere.cpp
        noa/gpu/cuda/filter/TestCUDACylinder.cpp
        noa/gpu/cuda/filter/TestCUDARectangle.cpp
        noa/gpu/cuda/filter/TestCUDAMedian.cpp
        noa/gpu/cuda/filter/TestCUDAConvolve.cpp

        noa/gpu/cuda/transform/TestCUDARotate.cpp
        noa/gpu/cuda/transform/TestCUDAScale.cpp
        noa/gpu/cuda/transform/TestCUDATranslate.cpp
        noa/gpu/cuda/transform/TestCUDAApply.cpp
        noa/gpu/cuda/transform/TestCUDAApplySymmetry.cpp
        noa/gpu/cuda/transform/TestCUDASymmetry.cpp

        noa/gpu/cuda/reconstruct/TestCUDAProjections.cpp

        )

if (NOA_ENABLE_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_tests.cmake)
