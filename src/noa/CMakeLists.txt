# Collect header and source files.
# Note: All of these file-paths should be relative to this directory!

include(${PROJECT_SOURCE_DIR}/src/noa/core/Append.cmake)
if (NOA_ENABLE_CPU)
    include(${PROJECT_SOURCE_DIR}/src/noa/cpu/Append.cmake)
endif ()
if (NOA_ENABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/src/noa/gpu/cuda/Append.cmake)
endif ()
#include(${PROJECT_SOURCE_DIR}/src/noa/unified/Append.cmake)

# Library sources.
# Note: use quotes in case these are empty.
set(NOA_SOURCES
    ${NOA_CORE_SOURCES}
    ${NOA_CPU_SOURCES}
    ${NOA_CUDA_SOURCES}
    ${NOA_UNIFIED_SOURCES}
)

# Library headers that should also be installed.
# Note: The entire project directory is already included for the build,
#       so these are to explicitly specify the headers that should be installed.
set(NOA_PUBLIC_HEADERS
    ${NOA_CORE_PUBLIC_HEADERS}
    ${NOA_CPU_PUBLIC_HEADERS}
    ${NOA_CUDA_PUBLIC_HEADERS}
    ${NOA_UNIFIED_PUBLIC_HEADERS}

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

# Create and set up the main target.
include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
