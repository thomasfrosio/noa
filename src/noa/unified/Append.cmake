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
        unified/FFT.h
        unified/Memory.h
        unified/Stream.h
        unified/Stream.inl

        # noa::fft
        unified/fft/Remap.h
        unified/fft/Remap.inl
        unified/fft/Resize.h
        unified/fft/Resize.inl
        unified/fft/Transform.h
        unified/fft/Transform.inl

        # noa::signal
        unified/signal/fft/Bandpass.h
        unified/signal/fft/Shift.h
        unified/signal/Convolve.h
        unified/signal/Median.h
        unified/signal/Shape.h

        # noa::geometry
        unified/geometry/fft/Symmetry.h
        unified/geometry/fft/Transform.h
        unified/geometry/Prefilter.h
        unified/geometry/Rotate.h
        unified/geometry/Scale.h
        unified/geometry/Shift.h
        unified/geometry/Symmetry.h
        unified/geometry/Transform.h

        # noa::memory
        unified/memory/Cast.h
        unified/memory/Cast.inl
        unified/memory/Copy.h
        unified/memory/Index.h
        unified/memory/Index.inl
        unified/memory/Factory.h
        unified/memory/Factory.inl
        unified/memory/Resize.h
        unified/memory/Resize.inl
        unified/memory/Transpose.h
        unified/memory/Transpose.inl

        # noa::math
        unified/math/fft/Standardize.h
        unified/math/fft/Standardize.inl
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
