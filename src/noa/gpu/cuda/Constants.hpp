#pragma once

#include "noa/core/Traits.hpp"

namespace noa::cuda {
    struct Constant {
        static constexpr u32 WARP_SIZE = 32;
    };

    struct Limits {
        static constexpr u32 MAX_THREADS = 1024;
        static constexpr u32 MAX_X_BLOCKS = (1U << 31) - 1U;
        static constexpr u32 MAX_YZ_BLOCKS = 65535;
    };
}
