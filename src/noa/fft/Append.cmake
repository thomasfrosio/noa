list(APPEND NOA_HEADERS
    noa/fft/core/Frequency.hpp
    noa/fft/core/Layout.hpp
    noa/fft/core/Transform.hpp

    noa/fft/Factory.hpp
    noa/fft/Remap.hpp
    noa/fft/Resize.hpp
    noa/fft/Transform.hpp
)
list(APPEND NOA_SOURCES
    noa/fft/Transform.cpp
)

if (NOA_ENABLE_CPU)
    list(APPEND NOA_HEADERS
        noa/fft/cpu/Plan.hpp
        noa/fft/cpu/Transform.hpp
    )
    list(APPEND NOA_SOURCES
        noa/fft/cpu/Plan.cpp
    )
endif ()

if (NOA_ENABLE_CUDA)
    list(APPEND NOA_HEADERS
        noa/fft/cuda/Plan.hpp
        noa/fft/cuda/Transform.hpp
    )
    list(APPEND NOA_SOURCES
        noa/fft/cuda/Plan.cpp
    )
endif ()
