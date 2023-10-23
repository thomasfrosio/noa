# Included files for noa/cpu:
if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(NOA_CPU_HEADERS
    # noa::cpu
    cpu/AllocatorHeap.hpp
    cpu/Copy.hpp
    cpu/Device.hpp
    cpu/Event.hpp
    cpu/EwiseBinary.hpp
    cpu/EwiseTrinary.hpp
    cpu/EwiseUnary.hpp
    cpu/Iwise.hpp
    cpu/ReduceBinary.hpp
    cpu/ReduceUnary.hpp
    cpu/Set.hpp
    cpu/Sort.hpp
    cpu/Stream.hpp

    # noa::cpu::fft
    cpu/fft/Plan.hpp
    cpu/fft/Transforms.hpp
    #    cpu/fft/Remap.hpp
    #    cpu/fft/Resize.hpp

#    # noa::cpu::math
    cpu/math/Blas.hpp
#    cpu/math/Complex.hpp
#    cpu/math/LinAlg.hpp
#    cpu/math/Random.hpp
#    cpu/math/Reduce.hpp

#    # noa::cpu::signal
#    cpu/signal/fft/Bandpass.hpp
#    cpu/signal/fft/Correlate.hpp
#    cpu/signal/fft/CTF.hpp
#    cpu/signal/fft/FSC.hpp
#    cpu/signal/fft/PhaseShift.hpp
#    cpu/signal/fft/Standardize.hpp
#    cpu/signal/Convolve.hpp
#    cpu/signal/Median.hpp
#
#    cpu/memory/Arange.hpp
#    cpu/memory/Cast.hpp
#    cpu/memory/Index.hpp
#    cpu/memory/Iota.hpp
#    cpu/memory/Linspace.hpp
#    cpu/memory/Permute.hpp
#    cpu/memory/Resize.hpp
#    cpu/memory/Subregion.hpp
#
#    # noa::cpu::geometry
#    cpu/geometry/fft/Polar.hpp
#    cpu/geometry/fft/Project.hpp
#    cpu/geometry/fft/Shape.hpp
#    cpu/geometry/fft/Transform.hpp
#    cpu/geometry/Polar.hpp
#    cpu/geometry/Prefilter.hpp
#    cpu/geometry/Shape.hpp
#    cpu/geometry/Transform.hpp

    )

set(NOA_CPU_SOURCES
    # noa::cpu
    cpu/Device.cpp

    # noa::cpu::fft
    cpu/fft/Plan.cpp
#    cpu/fft/Remap.cpp
#    cpu/fft/Resize.cpp

    # noa::cpu::math
    cpu/math/Blas.cpp
#    cpu/math/Random.cpp
#    cpu/math/Reduce.cpp
#    cpu/math/LinAlg.cpp
#
#    # noa::cpu::signal
#    cpu/signal/fft/Bandpass.cpp
#    cpu/signal/fft/Correlate.cpp
#    cpu/signal/fft/CorrelatePeak.cpp
#    cpu/signal/fft/CTF.cpp
#    cpu/signal/fft/FSC.cpp
#    cpu/signal/fft/PhaseShift.cpp
#    cpu/signal/fft/Standardize.cpp
#    cpu/signal/Convolve.cpp
#    cpu/signal/Median.cpp
#
#    # noa::cpu::memory
#    cpu/memory/Permute.cpp
#    cpu/memory/Resize.cpp
#    cpu/memory/Subregion.cpp
#
#    # noa::cpu::geometry
#    cpu/geometry/fft/Polar.cpp
#    cpu/geometry/fft/FourierExtract.cpp
#    cpu/geometry/fft/FourierInsertRasterize.cpp
#    cpu/geometry/fft/FourierInsertInterpolate.cpp
#    cpu/geometry/fft/FourierInsertExtract.cpp
#    cpu/geometry/fft/FourierGriddingCorrection.cpp
#    cpu/geometry/fft/Shape2D.cpp
#    cpu/geometry/fft/Shape3D.cpp
#    cpu/geometry/fft/Transform.cpp
#    cpu/geometry/Polar.cpp
#    cpu/geometry/Prefilter.cpp
#    cpu/geometry/Transform.cpp
    )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})
