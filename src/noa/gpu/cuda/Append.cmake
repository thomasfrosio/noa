set(NOA_CUDA_HEADERS
    # noa::cuda
    gpu/cuda/Allocators.hpp
    gpu/cuda/Blas.hpp
    gpu/cuda/Block.cuh
    gpu/cuda/Constants.hpp
    gpu/cuda/Copy.cuh
    gpu/cuda/Device.hpp
    gpu/cuda/Event.hpp
    gpu/cuda/Ewise.cuh
    gpu/cuda/Exception.hpp
    gpu/cuda/Iwise.cuh
    gpu/cuda/Median.cuh
    gpu/cuda/MemoryPool.hpp
    gpu/cuda/Permute.cuh
    gpu/cuda/Pointers.hpp
    gpu/cuda/ReduceAxesEwise.cuh
    gpu/cuda/ReduceAxesIwise.cuh
    gpu/cuda/ReduceEwise.cuh
    gpu/cuda/ReduceIwise.cuh
    gpu/cuda/Runtime.hpp
    gpu/cuda/Sort.cuh
    gpu/cuda/Stream.hpp
    gpu/cuda/Texture.hpp
    gpu/cuda/Types.hpp
    gpu/cuda/Version.hpp
    gpu/cuda/Warp.cuh

    # noa::cuda::fft
    gpu/cuda/fft/Plan.hpp
    gpu/cuda/fft/Transforms.hpp

    # noa::cuda::signal
    gpu/cuda/signal/Convolve.cuh
    gpu/cuda/signal/MedianFilter.cuh

    # noa::cuda::geometry
    gpu/cuda/geometry/Prefilter.cuh
    )

set(NOA_CUDA_SOURCES
    gpu/cuda/Device.cpp
    gpu/cuda/Blas.cu

    # noa::cuda::fft
    gpu/cuda/fft/Plan.cpp
    )
