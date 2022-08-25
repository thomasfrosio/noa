# Collect header and source files in NOA_HEADERS and NOA_SOURCES
set(NOA_HEADERS)
set(NOA_SOURCES)

include(${PROJECT_SOURCE_DIR}/src/noa/common/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/cpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/gpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
