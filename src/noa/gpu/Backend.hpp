#pragma once

namespace noa::gpu {
    class Backend {
    public:
        enum Type {
            NONE, CUDA
        };

        static constexpr Type type() noexcept {
            #ifdef NOA_ENABLE_CUDA
            return Type::CUDA;
            #else
            return Type::NONE;
            #endif
        }

        static constexpr bool null() noexcept { return type() == Type::NONE; }
        static constexpr bool cuda() noexcept { return type() == Type::CUDA; }
    };
}
