# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        EntryPoint.cpp
        noa/utils/TestString.cpp
        noa/utils/TestAssert.cpp
        noa/utils/TestTraits.cpp
        noa/utils/TestInput.cpp)

# ---------------------------------------------------------------------------------------
enable_testing()

# Fetch Catch2
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.1
)
FetchContent_MakeAvailable(Catch2)

# - test that Catch2 is available and works
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
        NOA_TEST_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/noa/fixtures/")

#include(CTest)
#include(${Catch2_SOURCE_DIR}/contrib/ParseAndAddCatchTests.cmake)
#ParseAndAddCatchTests(noa_tests)
