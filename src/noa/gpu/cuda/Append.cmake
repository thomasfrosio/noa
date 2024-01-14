set(NOA_CUDA_HEADERS
    # noa::cuda
    gpu/cuda/Device.hpp
    gpu/cuda/Event.hpp
    gpu/cuda/Ewise.hpp
    gpu/cuda/Exception.hpp
    gpu/cuda/Find.hpp
    gpu/cuda/Sort.hpp
    gpu/cuda/Stream.hpp
    gpu/cuda/Types.hpp

    gpu/cuda/EwiseUnary.hpp

    # noa::cuda::utils
    gpu/cuda/utils/Block.cuh
    gpu/cuda/utils/EwiseBinary.cuh
    gpu/cuda/utils/EwiseTrinary.cuh
    gpu/cuda/utils/EwiseUnary.cuh
    gpu/cuda/utils/Iwise.cuh
    gpu/cuda/utils/Pointers.hpp
    gpu/cuda/utils/ReduceUnary.cuh
    gpu/cuda/utils/ReduceBinary.cuh
    gpu/cuda/utils/Version.hpp
    gpu/cuda/utils/Warp.cuh

    # noa::cuda::memory
    gpu/cuda/memory/AllocatorArray.hpp
    gpu/cuda/memory/AllocatorDevice.hpp
    gpu/cuda/memory/AllocatorDevicePadded.hpp
    gpu/cuda/memory/AllocatorManaged.hpp
    gpu/cuda/memory/AllocatorPinned.hpp
    gpu/cuda/memory/AllocatorTexture.hpp
    gpu/cuda/memory/Copy.hpp
    gpu/cuda/memory/MemoryPool.hpp

    # noa::cuda::fft
    gpu/cuda/fft/Exception.hpp
    gpu/cuda/fft/Plan.hpp
#    gpu/cuda/fft/Remap.hpp
#    gpu/cuda/fft/Resize.hpp
    gpu/cuda/fft/Transforms.hpp

    # noa::cuda::math
    gpu/cuda/math/Blas.hpp


    )

set(NOA_CUDA_SOURCES
    gpu/cuda/Device.cpp
    gpu/cuda/EwiseUnary.cu

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
