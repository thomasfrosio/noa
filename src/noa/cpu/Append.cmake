set(NOA_CPU_HEADERS
    # noa::cpu
    cpu/Allocators.hpp
    cpu/Blas.hpp
    cpu/Copy.hpp
    cpu/CubicBSplinePrefilter.hpp
    cpu/Device.hpp
    cpu/Event.hpp
    cpu/Ewise.hpp
    cpu/Iwise.hpp
    cpu/Median.hpp
    cpu/Permute.hpp
    cpu/ReduceAxesEwise.hpp
    cpu/ReduceAxesIwise.hpp
    cpu/ReduceEwise.hpp
    cpu/ReduceIwise.hpp
    cpu/Set.hpp
    cpu/Sort.hpp
    cpu/Stream.hpp

    # noa::cpu::fft
    cpu/fft/Plan.hpp
    cpu/fft/Transforms.hpp

    # noa::cpu::signal
    cpu/signal/Convolve.hpp
    cpu/signal/MedianFilter.hpp
    )

set(NOA_CPU_SOURCES
    # noa::cpu
    cpu/Blas.cpp
    cpu/Device.cpp

    # noa::cpu::fft
    cpu/fft/Plan.cpp
    )
