# ---------------------------------------------------------------------------------------
# Manages the project tests
# ---------------------------------------------------------------------------------------
enable_testing()

# Fetch Catch2
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.1
)
#FetchContent_MakeAvailable(Catch2)
FetchContent_GetProperties(Catch2)
if (NOT Catch2_POPULATED)
    FetchContent_Populate(Catch2)
    add_subdirectory(${Catch2_SOURCE_DIR} ${Catch2_BINARY_DIR} EXCLUDE_FROM_ALL)
endif ()


# - test that Catch2 is available and works
add_executable(noa_tests main.cpp noa/utils/testString.cpp)
target_link_libraries(noa_tests
        PRIVATE
        project_options
        Noa::noa
        Catch2::Catch2
        )

include(CTest)
include(${Catch2_SOURCE_DIR}/contrib/ParseAndAddCatchTests.cmake)
ParseAndAddCatchTests(noa_tests)
