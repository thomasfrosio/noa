#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include "noa/common/Types.h"

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
        bool layered{}; // this could be extracted from the array
    };
}
