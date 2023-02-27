# Included files for noa/cpu:
if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(NOA_CPU_HEADERS
        # noa::cpu
        cpu/Device.hpp
        cpu/Event.hpp
        cpu/Ewise.hpp
        cpu/Find.hpp
        cpu/Reduce.hpp
        cpu/Sort.hpp
        cpu/Stream.hpp

        # noa::cpu::utils
        cpu/utils/Iwise.hpp
        cpu/utils/EwiseUnary.hpp
        cpu/utils/EwiseBinary.hpp
        cpu/utils/EwiseTrinary.hpp
        cpu/utils/ReduceUnary.hpp
        cpu/utils/ReduceBinary.hpp

        # noa::cpu::fft
        cpu/fft/Plan.hpp
        cpu/fft/Remap.hpp
        cpu/fft/Resize.hpp
        cpu/fft/Transforms.hpp

        # noa::cpu::math
        cpu/math/Blas.hpp
        cpu/math/Complex.hpp
        cpu/math/LinAlg.hpp
        cpu/math/Random.hpp
        cpu/math/Reduce.hpp

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.h
        cpu/signal/fft/Correlate.h
        cpu/signal/fft/FSC.h
        cpu/signal/fft/PhaseShift.hpp
        cpu/signal/fft/Standardize.h
        cpu/signal/Convolve.h
        cpu/signal/Median.h

        # noa::cpu::memory
        cpu/memory/Arange.hpp
        cpu/memory/Cast.hpp
        cpu/memory/Copy.hpp
        cpu/memory/Index.hpp
        cpu/memory/Iota.hpp
        cpu/memory/Linspace.hpp
        cpu/memory/Permute.hpp
        cpu/memory/PtrHost.hpp
        cpu/memory/Resize.hpp
        cpu/memory/Set.hpp
        cpu/memory/Subregion.hpp

        # noa::cpu::geometry
        cpu/geometry/fft/Polar.hpp
        cpu/geometry/fft/Project.hpp
        cpu/geometry/fft/Shape.hpp
        cpu/geometry/fft/Transform.hpp
        cpu/geometry/Polar.hpp
        cpu/geometry/Prefilter.hpp
        cpu/geometry/Shape.hpp
        cpu/geometry/Transform.hpp

        )

set(NOA_CPU_SOURCES
        # noa::cpu
        cpu/Device.cpp
        cpu/Find.cpp
        cpu/Sort.cpp

        # noa::cpu::fft
        cpu/fft/Plan.cpp
        cpu/fft/Remap.cpp
        cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Blas.cpp
        cpu/math/Random.cpp
        cpu/math/Reduce.cpp
        cpu/math/LinAlg.cpp

        # noa::cpu::signal
        cpu/signal/fft/Bandpass.cpp
        cpu/signal/fft/Correlate.cpp
        cpu/signal/fft/CorrelatePeak.cpp
        cpu/signal/fft/FSC.cpp
        cpu/signal/fft/PhaseShift.cpp
        cpu/signal/fft/Standardize.cpp
        cpu/signal/Convolve.cpp
        cpu/signal/Median.cpp

        # noa::cpu::memory
        cpu/memory/Permute.cpp
        cpu/memory/Resize.cpp
        cpu/memory/Subregion.cpp

        # noa::cpu::geometry
        cpu/geometry/fft/Polar.cpp
        cpu/geometry/fft/Project.cpp
        cpu/geometry/fft/Shape2D.cpp
        cpu/geometry/fft/Shape3D.cpp
        cpu/geometry/fft/Transform.cpp
        cpu/geometry/Polar.cpp
        cpu/geometry/Prefilter.cpp
        cpu/geometry/Transform.cpp

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})
