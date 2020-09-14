# Fetch googletest.
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
)
FetchContent_MakeAvailable(googletest)

# Test the core and/or the programs.
set(UNIT_UNDER_TEST_CORE ${CMAKE_SOURCE_DIR}/include)
set(UNIT_UNDER_TEST_PRGS ${CMAKE_SOURCE_DIR}/src)

#########
# Tests #
#########

# - test that gtest is available and works
add_executable(noa_gtest noa_gtest.cpp)
target_link_libraries(noa_gtest
        PRIVATE
        root_warnings
        gmock # contains gtest
        gmock_main
        )
target_include_directories(noa_gtest
        PRIVATE
        ${UNIT_UNDER_TEST_CORE}
        )

add_test(noa_tests noa_gtest)
