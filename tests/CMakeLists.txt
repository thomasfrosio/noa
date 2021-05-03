# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        EntryPoint.cpp
        Helpers.h

        noa/util/TestString.cpp
        noa/util/TestTraits.cpp
        noa/util/TestIntXFloatX.cpp
        noa/util/TestComplex.cpp

        noa/io/TestOS.cpp
        noa/io/TestIO.cpp
        noa/io/TestInputs.cpp
        noa/io/files/TestTextFile.cpp
        noa/io/files/TestMRCFile.cpp

        # CPU backend
        noa/cpu/TestCPUPtrHost.cpp

        noa/cpu/fourier/TestCPUTransforms.cpp
        noa/cpu/fourier/TestCPUResize.cpp
        noa/cpu/fourier/TestCPURemap.cpp
        noa/cpu/fourier/TestCPUFilters.cpp

        noa/cpu/math/TestCPUArithmetics.cpp
        noa/cpu/math/TestCPUReductions.cpp
        noa/cpu/math/TestCPUArithmeticsComposite.cpp
        noa/cpu/math/TestCPUBooleans.cpp
        noa/cpu/math/TestCPUGenerics.cpp
        noa/cpu/math/TestCPUIndexes.cpp

        noa/cpu/masks/TestCPUSphere.cpp
        noa/cpu/masks/TestCPURectangle.cpp
        noa/cpu/masks/TestCPUCylinder.cpp

        noa/cpu/memory/TestCPUResize.cpp
        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        noa/gpu/cuda/TestCUDADevice.cpp
        noa/gpu/cuda/TestCUDAPtrPinned.cpp
        noa/gpu/cuda/TestCUDAPtrDevice.cpp
        noa/gpu/cuda/TestCUDAPtrDevicePadded.cpp
        noa/gpu/cuda/TestCUDAPtrArray.cpp
        noa/gpu/cuda/TestCUDAMemory.cpp

        noa/gpu/cuda/fourier/TestCUDATransforms.cpp
        noa/gpu/cuda/fourier/TestCUDAResize.cpp
        noa/gpu/cuda/fourier/TestCUDARemap.cpp
        noa/gpu/cuda/fourier/TestCUDAFilters.cpp

        noa/gpu/cuda/math/TestCUDAArithmetics.cpp
        noa/gpu/cuda/math/TestCUDAArithmeticsComposite.cpp
        noa/gpu/cuda/math/TestCUDABooleans.cpp
        noa/gpu/cuda/math/TestCUDAGenerics.cpp
        noa/gpu/cuda/math/TestCUDAIndexes.cpp
        noa/gpu/cuda/math/TestCUDAReductions.cpp

        noa/gpu/cuda/masks/TestCUDASphere.cpp
        noa/gpu/cuda/masks/TestCUDACylinder.cpp
        noa/gpu/cuda/masks/TestCUDARectangle.cpp

        )

if (NOA_BUILD_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/ConfigTests.cmake)
