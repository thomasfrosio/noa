#pragma once

#include "noa/core/Config.hpp"

namespace noa::gpu {
    #ifdef NOA_ENABLE_CUDA
    using namespace noa::cuda;
    #endif

    class Backend {
    public:
        enum class Type { NONE, CUDA };
        using enum Type;

        static constexpr Type type() noexcept {
            #ifdef NOA_ENABLE_CUDA
            return CUDA;
            #else
            return NONE;
            #endif
        }

        static constexpr bool null() noexcept { return type() == NONE; }
        static constexpr bool cuda() noexcept { return type() == CUDA; }
    };
}
