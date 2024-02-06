set(NOA_CPU_HEADERS
    # noa::cpu
    cpu/AllocatorHeap.hpp
    cpu/Blas.hpp
    cpu/Copy.hpp
    cpu/Device.hpp
    cpu/Event.hpp
    cpu/Ewise.hpp
    cpu/Iwise.hpp
    cpu/Permute.hpp
    cpu/ReduceAxesEwise.hpp
    cpu/ReduceEwise.hpp
    cpu/ReduceIwise.hpp
    cpu/Set.hpp
    cpu/Sort.hpp
    cpu/Stream.hpp

    # noa::cpu::fft
    cpu/fft/Plan.hpp
    cpu/fft/Transforms.hpp

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
    cpu/Blas.cpp
    cpu/Device.cpp

    # noa::cpu::fft
    cpu/fft/Plan.cpp

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
