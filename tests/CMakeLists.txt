enable_testing()

# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        EntryPoint.cpp
        Helpers.h

        noa/util/TestString.cpp
        noa/util/TestTraits.cpp
        noa/util/TestOS.cpp
        noa/util/TestIO.cpp
        noa/util/TestIntXFloatX.cpp
        noa/util/TestComplex.cpp
        noa/util/TestInputs.cpp

        noa/util/files/TestTextFile.cpp
        noa/util/files/TestMRCFile.cpp

        noa/cpu/TestPtrHost.cpp

        noa/cpu/fourier/TestTransforms.cpp
        noa/cpu/fourier/TestResize.cpp
        noa/cpu/fourier/TestRemap.cpp

        )

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        noa/gpu/cuda/TestDevice.cpp
        noa/gpu/cuda/TestPtrPinned.cpp
        noa/gpu/cuda/TestPtrDevice.cpp
        noa/gpu/cuda/TestPtrDevicePadded.cpp
        noa/gpu/cuda/TestPtrArray.cpp
        noa/gpu/cuda/TestMemory.cpp

        noa/gpu/cuda/fourier/TestTransforms.cpp

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
