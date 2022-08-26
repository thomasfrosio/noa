# Collect header and source files in NOA_HEADERS and NOA_SOURCES
set(NOA_SOURCES)
set(NOA_HEADERS
        FFT.h
        Geometry.h
        IO.h
        Logger.h
        Math.h
        Memory.h
        OS.h
        Session.h
        Signal.h
        String.h
        Types.h
        )

include(${PROJECT_SOURCE_DIR}/src/noa/common/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/cpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/gpu/Append.cmake)
include(${PROJECT_SOURCE_DIR}/src/noa/unified/Append.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
