set(NOA_CUDA_HEADERS
    # noa::cuda
    gpu/cuda/AllocatorArray.hpp
    gpu/cuda/AllocatorDevice.hpp
    gpu/cuda/AllocatorDevicePadded.hpp
    gpu/cuda/AllocatorManaged.hpp
    gpu/cuda/AllocatorPinned.hpp
    gpu/cuda/AllocatorTexture.hpp
    gpu/cuda/Blas.hpp
    gpu/cuda/Copy.cuh
    gpu/cuda/Device.hpp
    gpu/cuda/Event.hpp
    gpu/cuda/Ewise.cuh
    gpu/cuda/Exception.hpp
    gpu/cuda/Iwise.cuh
    gpu/cuda/MemoryPool.hpp
    gpu/cuda/Permute.cuh
    gpu/cuda/Pointers.hpp
    gpu/cuda/ReduceAxesEwise.cuh
    gpu/cuda/ReduceAxesIwise.cuh
    gpu/cuda/ReduceEwise.cuh
    gpu/cuda/ReduceIwise.cuh
    gpu/cuda/Sort.cuh
    gpu/cuda/Stream.hpp
    gpu/cuda/Types.hpp
    gpu/cuda/Version.hpp

    # kernels
    gpu/cuda/kernels/Block.cuh
    gpu/cuda/kernels/Ewise.cuh
    gpu/cuda/kernels/Iwise.cuh
    gpu/cuda/kernels/ReduceAxesEwise.cuh
    gpu/cuda/kernels/ReduceAxesIwise.cuh
    gpu/cuda/kernels/ReduceEwise.cuh
    gpu/cuda/kernels/ReduceIwise.cuh
    gpu/cuda/kernels/Warp.cuh
    gpu/cuda/kernels/Permute.cuh

    # noa::cuda::fft
    gpu/cuda/fft/Exception.hpp
    gpu/cuda/fft/Plan.hpp
    gpu/cuda/fft/Transforms.hpp
    )

set(NOA_CUDA_SOURCES
    gpu/cuda/Device.cpp
    gpu/cuda/Blas.cu

    # noa::cuda::fft
    gpu/cuda/fft/Exception.cpp
    gpu/cuda/fft/Plan.cpp
    )
