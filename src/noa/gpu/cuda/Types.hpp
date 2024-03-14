#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include "noa/core/Types.hpp"

#ifdef NOA_IS_OFFLINE
#include <memory>
#endif

namespace noa::cuda {
    struct LaunchConfig {
        dim3 n_blocks;
        dim3 n_threads;
        size_t n_bytes_of_shared_memory{};
        bool is_cooperative{};
    };

    struct Constant {
        static constexpr u32 WARP_SIZE = 32;
    };

    struct Limits {
        static constexpr u32 MAX_THREADS = 1024;
        static constexpr u32 MAX_X_BLOCKS = (1U << 31) - 1U;
        static constexpr u32 MAX_YZ_BLOCKS = 65535;
    };
}
