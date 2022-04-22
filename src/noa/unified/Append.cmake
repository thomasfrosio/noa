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

        # noa::fft
        unified/fft/Filter.h
        unified/fft/Remap.h
        unified/fft/Resize.h
        unified/fft/Transform.h

        # noa::filter
        unified/filter/Convolve.h
        unified/filter/Median.h
        unified/filter/Shape.h

        # noa::geometry
        unified/geometry/fft/Shift.h
        unified/geometry/fft/Symmetry.h
        unified/geometry/fft/Transform.h
        unified/geometry/Prefilter.h
        unified/geometry/Rotate.h
        unified/geometry/Scale.h
        unified/geometry/Shift.h
        unified/geometry/Symmetry.h
        unified/geometry/Transform.h

        # noa::memory
        unified/memory/Initialize.h
        unified/memory/Cast.h
        unified/memory/Copy.h
        unified/memory/Index.h
        unified/memory/Resize.h
        unified/memory/Transpose.h

        # noa::math
        unified/math/Complex.h
        unified/math/Ewise.h
        unified/math/Random.h
        unified/math/Reduce.h
        )

set(NOA_UNIFIED_SOURCES
        unified/Stream.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
