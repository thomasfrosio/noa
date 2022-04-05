# Included files for noa/unified:

set(NOA_UNIFIED_HEADERS
        unified/Array.h
        unified/Array.inl
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
