# Included files for noa/unified:
if (NOT NOA_ENABLE_UNIFIED)
    return()
endif ()

set(NOA_UNIFIED_HEADERS
        unified/Allocator.h
        unified/Array.h
        unified/ArrayOption.h
        unified/Device.h
        unified/Device.inl
        unified/Stream.h
        unified/Stream.inl

        # noa::memory
        unified/memory/Initialize.h
        unified/memory/Cast.h
        unified/memory/Copy.h
        unified/memory/Index.h
        unified/memory/Resize.h
        unified/memory/Transpose.h
        )

set(NOA_UNIFIED_SOURCES
        unified/Array.cpp
        unified/Stream.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
