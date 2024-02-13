set(NOA_CUDA_HEADERS
    # noa::cuda
    gpu/cuda/AllocatorArray.hpp
    gpu/cuda/AllocatorDevice.hpp
    gpu/cuda/AllocatorDevicePadded.hpp
    gpu/cuda/AllocatorManaged.hpp
    gpu/cuda/AllocatorPinned.hpp
    gpu/cuda/AllocatorTexture.hpp
    gpu/cuda/Device.hpp
    gpu/cuda/Event.hpp
    gpu/cuda/Exception.hpp
    gpu/cuda/MemoryPool.hpp
    gpu/cuda/Pointers.hpp
    gpu/cuda/Stream.hpp
    gpu/cuda/Types.hpp
    gpu/cuda/Version.hpp

    gpu/cuda/Ewise.hpp
    gpu/cuda/Iwise.hpp
    gpu/cuda/ReduceAxesEwise.hpp
    gpu/cuda/ReduceEwise.hpp
    gpu/cuda/ReduceIwise.hpp

    #    gpu/cuda/Sort.hpp
    #    gpu/cuda/Copy.hpp
    #    gpu/cuda/Blas.hpp

    # kernels
    gpu/cuda/kernels/Block.cuh
    gpu/cuda/kernels/Ewise.cuh
    gpu/cuda/kernels/Iwise.cuh
    gpu/cuda/kernels/ReduceAxesEwise.cuh
    gpu/cuda/kernels/ReduceEwise.cuh
    gpu/cuda/kernels/ReduceIwise.cuh
    gpu/cuda/kernels/Warp.cuh

    # noa::cuda::fft
    gpu/cuda/fft/Exception.hpp
    gpu/cuda/fft/Plan.hpp
    gpu/cuda/fft/Transforms.hpp
    )

set(NOA_CUDA_SOURCES
    gpu/cuda/Device.cpp

    # noa::cuda::fft
    gpu/cuda/fft/Exception.cpp
    gpu/cuda/fft/Plan.cpp

    )

# Files to be preprocessed and runtime-compiled.
set(NOA_CUDA_PREPROCESS_SOURCES
    gpu/cuda/utils/EwiseBinary.cuh
    gpu/cuda/utils/EwiseTrinary.cuh
    gpu/cuda/utils/EwiseUnary.cuh
)
