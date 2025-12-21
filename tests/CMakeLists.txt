set(TEST_SOURCES
    Assets.hpp
    Catch.hpp
    EntryPoint.cpp
    Utils.hpp
)

include(${PROJECT_SOURCE_DIR}/tests/noa/base/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/runtime/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/fft/Append.cmake)
include(${PROJECT_SOURCE_DIR}/tests/noa/io/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/tests/noa/signal/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/tests/noa/xform/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa_tests.cmake)
