#pragma once

namespace noa::gpu {
    class Backend {
    public:
        enum Type {
            NONE, CUDA, VULKAN, METAL
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
        static constexpr bool vulkan() noexcept { return type() == Type::VULKAN; }
        static constexpr bool metal() noexcept { return type() == Type::METAL; }
    };
}
