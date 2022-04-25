# Collect header and source files in NOA_HEADERS and NOA_SOURCES
set(NOA_HEADERS Session.h)
set(NOA_SOURCES Session.cpp)

include(${PROJECT_SOURCE_DIR}/src/noa/common/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/cpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/gpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/unified/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
