# Collect source files for tests:
set(TEST_SOURCES
    Assets.h
    EntryPoint.cpp
    Utils.hpp
    )
include(${PROJECT_SOURCE_DIR}/tests/noa/core/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/cpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/gpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/unified/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_tests.cmake)
