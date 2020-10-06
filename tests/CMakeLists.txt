# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
set(TEST_SOURCES
        EntryPoint.cpp
        noa/utils/TestString.cpp
        noa/utils/TestAssert.cpp
        noa/utils/TestTraits.cpp
        noa/utils/TestInputs.cpp)

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
#FetchContent_GetProperties(Catch2)
#if (NOT Catch2_POPULATED)
#    FetchContent_Populate(Catch2)
#    add_subdirectory(${Catch2_SOURCE_DIR} ${Catch2_BINARY_DIR})
#endif ()


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

include(CTest)
include(${Catch2_SOURCE_DIR}/contrib/ParseAndAddCatchTests.cmake)
ParseAndAddCatchTests(noa_tests)
