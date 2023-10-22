# Collect header and source files in NOA_HEADERS and NOA_SOURCES
set(NOA_SOURCES)
set(NOA_HEADERS
#    Array.hpp
#    FFT.hpp
#    Geometry.hpp
#    IO.hpp
#    Math.hpp
#    Memory.hpp
#    Session.hpp
#    Signal.hpp
#    String.hpp
    )

include(${PROJECT_SOURCE_DIR}/src/noa/core/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/src/noa/algorithms/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/cpu/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/src/noa/gpu/Append.cmake)
#include(${PROJECT_SOURCE_DIR}/src/noa/unified/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
