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
        noa/util/TestVectors.cpp

        noa/managers/TestInputManager.cpp

        noa/files/TestTextFile.cpp
        noa/files/TestImageFile.cpp

        )

# ---------------------------------------------------------------------------------------
enable_testing()

# Fetch Catch2
include(${PROJECT_SOURCE_DIR}/ext/catch2/catch2.cmake)

add_executable(noa_tests ${TEST_SOURCES})
target_link_libraries(noa_tests
        PRIVATE
        project_options
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
