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
        unified/Geometry.h
        unified/IO.h
        unified/Math.h
        unified/Memory.h
        unified/Signal.h
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
        unified/signal/fft/Bandpass.inl
        unified/signal/fft/Correlate.h
        unified/signal/fft/Correlate.inl
        unified/signal/fft/Shift.h
        unified/signal/fft/Shift.inl
        unified/signal/fft/Standardize.h
        unified/signal/fft/Standardize.inl
        unified/signal/Convolve.h
        unified/signal/Convolve.inl
        unified/signal/Median.h
        unified/signal/Median.inl
        unified/signal/Shape.h
        unified/signal/Shape.inl

        # noa::geometry
        unified/geometry/fft/Polar.h
        unified/geometry/fft/Polar.inl
        unified/geometry/fft/Project.h
        unified/geometry/fft/Project.inl
        unified/geometry/fft/Symmetry.h
        unified/geometry/fft/Symmetry.inl
        unified/geometry/fft/Transform.h
        unified/geometry/fft/Transform.inl
        unified/geometry/Polar.h
        unified/geometry/Polar.inl
        unified/geometry/Prefilter.h
        unified/geometry/Prefilter.inl
        unified/geometry/Rotate.h
        unified/geometry/Rotate.inl
        unified/geometry/Scale.h
        unified/geometry/Scale.inl
        unified/geometry/Shift.h
        unified/geometry/Shift.inl
        unified/geometry/Symmetry.h
        unified/geometry/Symmetry.inl
        unified/geometry/Transform.h
        unified/geometry/Transform.inl

        # noa::memory
        unified/memory/Cast.h
        unified/memory/Cast.inl
        unified/memory/Copy.h
        unified/memory/Factory.h
        unified/memory/Factory.inl
        unified/memory/Index.h
        unified/memory/Index.inl
        unified/memory/Permute.h
        unified/memory/Permute.inl
        unified/memory/Resize.h
        unified/memory/Resize.inl

        # noa::math
        unified/math/Blas.h
        unified/math/Blas.inl
        unified/math/Complex.h
        unified/math/Complex.inl
        unified/math/Ewise.h
        unified/math/Ewise.inl
        unified/math/Find.h
        unified/math/Find.inl
        unified/math/LinAlg.h
        unified/math/LinAlg.inl
        unified/math/Random.h
        unified/math/Random.inl
        unified/math/Reduce.h
        unified/math/Reduce.inl
        )

set(NOA_UNIFIED_SOURCES
        unified/Device.cpp
        unified/Stream.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
