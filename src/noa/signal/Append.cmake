list(APPEND NOA_HEADERS
    noa/signal/core/Correlation.hpp
    noa/signal/core/CTF.hpp
    noa/signal/core/LeastSquare.hpp
    noa/signal/core/Window.hpp

    noa/signal/Bandpass.hpp
    noa/signal/Convolve.hpp
    noa/signal/Correlate.hpp
    noa/signal/CTF.hpp
    noa/signal/FilterSpectrum.hpp
    noa/signal/FSC.hpp
    noa/signal/MedianFilter.hpp
    noa/signal/PhaseShift.hpp
    noa/signal/Standardize.hpp
    noa/signal/Traits.hpp
    noa/signal/Window.hpp
)

if (NOA_ENABLE_CPU)
    list(APPEND NOA_HEADERS
        noa/signal/cpu/Convolve.hpp
        noa/signal/cpu/MedianFilter.hpp
    )
endif ()

if (NOA_ENABLE_CUDA)
    list(APPEND NOA_HEADERS
        noa/signal/cuda/Convolve.cuh
        noa/signal/cuda/MedianFilter.cuh
    )
endif ()
