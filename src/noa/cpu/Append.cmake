# Included files for noa/cpu:
if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(NOA_CPU_HEADERS
        # noa::cpu
        cpu/Device.h
        cpu/Event.h
        cpu/Stream.h

        # # noa::cpu::fft
        cpu/fft/Plan.h
        cpu/fft/Remap.h
        cpu/fft/Resize.h
        cpu/fft/Transforms.h

        # noa::cpu::math
        cpu/math/Blas.h
        cpu/math/Complex.h
        cpu/math/Ewise.h
        cpu/math/Ewise.inl
        cpu/math/Find.h
        cpu/math/LinAlg.h
        cpu/math/Random.h
        cpu/math/Reduce.h
        cpu/math/Reduce.inl

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.h
        cpu/signal/fft/Correlate.h
        cpu/signal/fft/Shift.h
        cpu/signal/fft/Standardize.h
        cpu/signal/Convolve.h
        cpu/signal/Median.h
        cpu/signal/Shape.h
        cpu/signal/Window.h

        # noa::cpu::memory
        cpu/memory/Arange.h
        cpu/memory/Cast.h
        cpu/memory/Copy.h
        cpu/memory/Index.h
        cpu/memory/Index.inl
        cpu/memory/Iota.h
        cpu/memory/Linspace.h
        cpu/memory/Permute.h
        cpu/memory/Permute.inl
        cpu/memory/PtrHost.h
        cpu/memory/Resize.h
        cpu/memory/Set.h

        # noa::cpu::geometry
        cpu/geometry/fft/Polar.h
        cpu/geometry/fft/Project.h
        cpu/geometry/fft/Symmetry.h
        cpu/geometry/fft/Transform.h
        cpu/geometry/Interpolate.h
        cpu/geometry/Interpolator.h
        cpu/geometry/Polar.h
        cpu/geometry/Prefilter.h
        cpu/geometry/Rotate.h
        cpu/geometry/Scale.h
        cpu/geometry/Shift.h
        cpu/geometry/Symmetry.h
        cpu/geometry/Transform.h

        )

set(NOA_CPU_SOURCES
        cpu/Device.cpp

        # # noa::cpu::fft
        cpu/fft/Plan.cpp
        cpu/fft/Remap.cpp
        cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Blas.cpp
        cpu/math/Find.cpp
        cpu/math/Random.cpp
        cpu/math/Reduce.cpp
        cpu/math/LinAlg.cpp

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.cpp
        cpu/signal/fft/Correlate.cpp
        cpu/signal/fft/Shift.cpp
        cpu/signal/fft/Standardize.cpp
        cpu/signal/Convolve.cpp
        cpu/signal/Median.cpp
        cpu/signal/ShapeCylinder.cpp
        cpu/signal/ShapeRectangle.cpp
        cpu/signal/ShapeSphere.cpp

        # noa::cpu::memory
        cpu/memory/Index.cpp
        cpu/memory/Permute.cpp
        cpu/memory/Resize.cpp

        # noa::cpu::geometry
        cpu/geometry/fft/Polar.cpp
        cpu/geometry/fft/Project.cpp
        cpu/geometry/fft/Transform.cpp
        cpu/geometry/fft/TransformSymmetry.cpp
        cpu/geometry/Polar.cpp
        cpu/geometry/Prefilter.cpp
        cpu/geometry/Shift.cpp
        cpu/geometry/Symmetry.cpp
        cpu/geometry/Transform.cpp
        cpu/geometry/TransformSymmetry.cpp

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})
