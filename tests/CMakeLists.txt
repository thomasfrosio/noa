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


        )

# ---------------------------------------------------------------------------------------
enable_testing()

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
set(TEST_CUDA_SOURCES
        noa/gpu/cuda/TestDevice.cpp


        )

if (NOA_BUILD_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CUDA_SOURCES})
endif ()

# Fetch Catch2
include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)

add_executable(noa_tests ${TEST_SOURCES})
target_link_libraries(noa_tests
        PRIVATE
        prj_compiler_option
        Noa::noa
        Catch2::Catch2
        )

target_precompile_headers(noa_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/noa/Base.h)

target_compile_definitions(noa_tests
        PRIVATE
        NOA_TESTS_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

#include(CTest)
#include(${Catch2_SOURCE_DIR}/contrib/ParseAndAddCatchTests.cmake)
#ParseAndAddCatchTests(noa_tests)
