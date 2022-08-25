# Included files for noa/unified:
if (NOT NOA_ENABLE_UNIFIED)
    return()
endif ()

set(NOA_UNIFIED_HEADERS
        Allocator.h
        Array.h
        ArrayOption.h
        Device.h
        FFT.h
        Geometry.h
        IO.h
        Logger.h
        Math.h
        Memory.h
        OS.h
        Session.h
        Signal.h
        Stream.h
        String.h
        Texture.h
        Types.h
        details/Array.inl
        details/Device.inl
        details/Stream.inl
        details/Texture.inl

        # noa::fft
        fft/Remap.h
        fft/Resize.h
        fft/Transform.h
        fft/details/Remap.inl
        fft/details/Resize.inl
        fft/details/Transform.inl

        # noa::signal
        signal/fft/Bandpass.h
        signal/fft/Correlate.h
        signal/fft/Shift.h
        signal/fft/Standardize.h
        signal/fft/details/Bandpass.inl
        signal/fft/details/Correlate.inl
        signal/fft/details/Shift.inl
        signal/fft/details/Standardize.inl
        signal/Convolve.h
        signal/Median.h
        signal/Shape.h
        signal/details/Convolve.inl
        signal/details/Median.inl
        signal/details/Shape.inl

        # noa::geometry
        geometry/fft/Polar.h
        geometry/fft/Project.h
        geometry/fft/Symmetry.h
        geometry/fft/Transform.h
        geometry/fft/details/Polar.inl
        geometry/fft/details/Project.inl
        geometry/fft/details/Symmetry.inl
        geometry/fft/details/Transform.inl
        geometry/Polar.h
        geometry/Prefilter.h
        geometry/Rotate.h
        geometry/Scale.h
        geometry/Shift.h
        geometry/Symmetry.h
        geometry/Transform.h
        geometry/details/Polar.inl
        geometry/details/Prefilter.inl
        geometry/details/Rotate.inl
        geometry/details/Scale.inl
        geometry/details/Shift.inl
        geometry/details/Symmetry.inl
        geometry/details/Transform.inl

        # noa::memory
        memory/Cast.h
        memory/Copy.h
        memory/Factory.h
        memory/Index.h
        memory/Permute.h
        memory/Resize.h
        memory/details/Cast.inl
        memory/details/Factory.inl
        memory/details/Index.inl
        memory/details/Permute.inl
        memory/details/Resize.inl

        # noa::math
        math/Blas.h
        math/Complex.h
        math/Ewise.h
        math/Find.h
        math/LinAlg.h
        math/Random.h
        math/Reduce.h
        math/details/Blas.inl
        math/details/Complex.inl
        math/details/Ewise.inl
        math/details/Find.inl
        math/details/LinAlg.inl
        math/details/Random.inl
        math/details/Reduce.inl
        )

set(NOA_UNIFIED_SOURCES
        Device.cpp
        Stream.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_UNIFIED_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_UNIFIED_SOURCES})
