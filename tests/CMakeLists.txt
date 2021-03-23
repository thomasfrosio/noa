enable_testing()

# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        EntryPoint.cpp
        Helpers.h

        src/noa/util/TestString.cpp
        src/noa/util/TestTraits.cpp
        src/noa/util/TestIntXFloatX.cpp
        src/noa/util/TestComplex.cpp

        src/noa/io/TestOS.cpp
        src/noa/io/TestIO.cpp
        src/noa/io/TestInputs.cpp
        src/noa/io/files/TestTextFile.cpp
        src/noa/io/files/TestMRCFile.cpp

        # CPU backend
        src/noa/cpu/fourier/TestCPUTransforms.cpp
        src/noa/cpu/fourier/TestCPUResize.cpp
        src/noa/cpu/fourier/TestCPURemap.cpp
        src/noa/cpu/math/TestCPUArithmetics.cpp
        src/noa/cpu/TestCPUPtrHost.cpp

        src/noa/cpu/math/TestCPUReductions.cpp)

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        src/noa/gpu/cuda/TestCUDADevice.cpp
        src/noa/gpu/cuda/TestCUDAPtrPinned.cpp
        src/noa/gpu/cuda/TestCUDAPtrDevice.cpp
        src/noa/gpu/cuda/TestCUDAPtrDevicePadded.cpp
        src/noa/gpu/cuda/TestCUDAPtrArray.cpp
        src/noa/gpu/cuda/TestCUDAMemory.cpp

        src/noa/gpu/cuda/fourier/TestCUDATransforms.cpp
        src/noa/gpu/cuda/fourier/TestCUDAResize.cpp
        src/noa/gpu/cuda/fourier/TestCUDARemap.cpp

        )

if (NOA_BUILD_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
endif ()

# ---------------------------------------------------------------------------------------
# Built target
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)

add_executable(noa_tests ${TEST_SOURCES})
target_link_libraries(noa_tests
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        Noa::noa
        Catch2::Catch2
        )

target_precompile_headers(noa_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/Definitions.h
        ${PROJECT_SOURCE_DIR}/src/noa/Errno.h
        ${PROJECT_SOURCE_DIR}/src/noa/Exception.h
        ${PROJECT_SOURCE_DIR}/src/noa/Log.h
        ${PROJECT_SOURCE_DIR}/src/noa/Types.h
        )

target_compile_definitions(noa_tests
        PRIVATE
        NOA_TESTS_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Global includes. Used by all targets linking to the library (PUBLIC)
# Headers in source tree can be included as well as the noa_generated_headers
target_include_directories(noa_tests
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests>"
        "$<INSTALL_INTERFACE:.>"
        )

#include(CTest)
#include(${Catch2_SOURCE_DIR}/contrib/ParseAndAddCatchTests.cmake)
#ParseAndAddCatchTests(noa_tests)
