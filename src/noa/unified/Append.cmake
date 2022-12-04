# Included files for noa/unified:
if (NOT NOA_ENABLE_UNIFIED)
    return()
endif ()

set(NOA_UNIFIED_HEADERS
        unified/Allocator.h
        unified/Array.h
        unified/Array.inl
        unified/ArrayOption.h
        unified/Device.h
        unified/Device.inl
        unified/Stream.h
        unified/Stream.inl
        unified/Texture.h
        unified/Texture.inl

        # noa::io
        unified/io/ImageFile.h
        unified/io/ImageFile.inl

        # noa::fft
        unified/fft/Factory.h
        unified/fft/Factory.inl
        unified/fft/Remap.h
        unified/fft/Remap.inl
        unified/fft/Resize.h
        unified/fft/Resize.inl
        unified/fft/Transform.h
        unified/fft/Transform.inl

        # noa::signal
        unified/signal/Convolve.h
        unified/signal/Convolve.inl
        unified/signal/fft/Bandpass.h
        unified/signal/fft/Bandpass.inl
        unified/signal/fft/Correlate.h
        unified/signal/fft/Shape.h
        unified/signal/fft/Shape.inl
        unified/signal/fft/Shift.h
        unified/signal/fft/Shift.inl
        unified/signal/fft/Standardize.h
        unified/signal/fft/Standardize.inl
        unified/signal/Median.h
        unified/signal/Median.inl
        unified/signal/Shape.h
        unified/signal/Shape.inl

        # noa::geometry
        unified/geometry/fft/Polar.h
        unified/geometry/fft/Project.h
        unified/geometry/fft/Transform.h
        unified/geometry/Polar.h
        unified/geometry/Prefilter.h
        unified/geometry/Prefilter.inl
        unified/geometry/Transform.h

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
        unified/math/Sort.h
        unified/math/Sort.inl
        )

set(NOA_UNIFIED_SOURCES
        unified/Device.cpp
        unified/Stream.cpp

        # noa::geometry
        unified/geometry/Polar.cpp
        unified/geometry/Transform.cpp
        unified/geometry/fft/Polar.cpp
        unified/geometry/fft/Project.cpp
        unified/geometry/fft/Transform.cpp

        unified/signal/fft/Correlate.cpp

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
