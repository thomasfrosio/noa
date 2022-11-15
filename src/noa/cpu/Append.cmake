# Included files for noa/cpu:
if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(NOA_CPU_HEADERS
        # noa::cpu
        cpu/Device.h
        cpu/Event.h
        cpu/Stream.h

        # noa::cpu::utils
        cpu/utils/Iwise.h

        # noa::cpu::fft
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
        cpu/math/Sort.h

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.h
        cpu/signal/fft/Correlate.h
        cpu/signal/fft/Shape.h
        cpu/signal/fft/Shift.h
        cpu/signal/fft/Standardize.h
        cpu/signal/Convolve.h
        cpu/signal/Median.h
        cpu/signal/Shape.h

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
        cpu/geometry/fft/Transform.h
        cpu/geometry/Polar.h
        cpu/geometry/Prefilter.h
        cpu/geometry/Transform.h

        )

set(NOA_CPU_SOURCES
        # noa::cpu
        cpu/Device.cpp

        # noa::cpu::fft
        cpu/fft/Plan.cpp
        cpu/fft/Remap.cpp
        cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Blas.cpp
        cpu/math/Find.cpp
        cpu/math/Random.cpp
        cpu/math/Reduce.cpp
        cpu/math/LinAlg.cpp
        cpu/math/Sort.cpp

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.cpp
        cpu/signal/fft/Correlate.cpp
        cpu/signal/fft/CorrelatePeak.cpp
        cpu/signal/fft/Shape.cpp
        cpu/signal/fft/Shift.cpp
        cpu/signal/fft/Standardize.cpp
        cpu/signal/Convolve.cpp
        cpu/signal/Median.cpp

        # noa::cpu::memory
        cpu/memory/Index.cpp
        cpu/memory/Permute.cpp
        cpu/memory/Resize.cpp

        # noa::cpu::geometry
        cpu/geometry/fft/Polar.cpp
        cpu/geometry/fft/Project.cpp
        cpu/geometry/fft/Transform.cpp
        cpu/geometry/Polar.cpp
        cpu/geometry/Prefilter.cpp
        cpu/geometry/Transform.cpp

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})
