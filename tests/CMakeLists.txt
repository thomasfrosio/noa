# Collect source files for tests:
set(TEST_SOURCES
        Assets.h
        EntryPoint.cpp
        Helpers.h
        )
include(${PROJECT_SOURCE_DIR}/tests/noa/common/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/cpu/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/tests/noa/gpu/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/tests/noa/unified/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_tests.cmake)
