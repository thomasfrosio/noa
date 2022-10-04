#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "noa/common/Types.h"

// Add specialization for cfloat_t, which has the same layout as float2 (struct of 2 floats).
// Mostly used for CUDA arrays (i.e. PtrArray) and textures (i.e. PtrTexture).
template<>
__inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<noa::cfloat_t>() {
    return cudaCreateChannelDesc<float2>();
}

// Ensure BorderMode and InterpMode are compatible with cudaTextureAddressMode and cudaTextureFilterMode.
// This is used by noa::cuda::memory::PtrTexture.
static_assert(noa::BORDER_PERIODIC == static_cast<int>(cudaAddressModeWrap));
static_assert(noa::BORDER_CLAMP == static_cast<int>(cudaAddressModeClamp));
static_assert(noa::BORDER_MIRROR == static_cast<int>(cudaAddressModeMirror));
static_assert(noa::BORDER_ZERO == static_cast<int>(cudaAddressModeBorder));
static_assert(noa::INTERP_NEAREST == static_cast<int>(cudaFilterModePoint));
static_assert(noa::INTERP_LINEAR == static_cast<int>(cudaFilterModeLinear));

namespace noa::cuda {
    struct LaunchConfig {
        dim3 blocks;
        dim3 threads;
        size_t bytes_shared_memory{};
        bool cooperative{};
    };

    struct Limits {
        static constexpr uint32_t WARP_SIZE = 32;
        static constexpr uint32_t MAX_THREADS = 1024;
        static constexpr uint32_t MAX_X_BLOCKS = (1U << 31) - 1U;
        static constexpr uint32_t MAX_YZ_BLOCKS = 65535;
    };

    template<typename T>
    struct Texture {
        std::shared_ptr<cudaArray> array{nullptr};
        std::shared_ptr<cudaTextureObject_t> texture{};
    };
}
