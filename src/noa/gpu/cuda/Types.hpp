#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include "noa/core/Types.hpp"

namespace noa::cuda {
    struct LaunchConfig {
        dim3 blocks;
        dim3 threads;
        size_t bytes_shared_memory{};
        bool cooperative{};
    };

    struct Constant {
        static constexpr u32 WARP_SIZE = 32;
    };

    struct Limits {
        static constexpr u32 MAX_THREADS = 1024;
        static constexpr u32 MAX_X_BLOCKS = (1U << 31) - 1U;
        static constexpr u32 MAX_YZ_BLOCKS = 65535;
    };

    template<typename T>
    struct Texture {
        Shared<cudaArray> array{nullptr};
        Shared<cudaTextureObject_t> texture{};
    };

    template<typename T>
    struct TextureObject {
        using value_type = T;
        cudaArray_t array{nullptr};
        cudaTextureObject_t texture{};
        InterpMode interp_mode{};
    };
}
