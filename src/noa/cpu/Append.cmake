set(NOA_CPU_HEADERS
    # noa::cpu
    cpu/AllocatorHeap.hpp
    cpu/Blas.hpp
    cpu/Copy.hpp
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

    # noa::cuda::signal
    cpu/signal/Convolve.hpp
    cpu/signal/Correlate.hpp
    )

set(NOA_CPU_SOURCES
    # noa::cpu
    cpu/Blas.cpp
    cpu/Device.cpp

    # noa::cpu::fft
    cpu/fft/Plan.cpp
    )
