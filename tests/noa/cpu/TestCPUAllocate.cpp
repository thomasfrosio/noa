#include <noa/cpu/AllocatorHeap.hpp>

#include "Catch.hpp"

TEST_CASE("cpu::AllocatorHeap") {
    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap<float>::allocate(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 256));
    }

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap<float>::allocate<512>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 512));
    }

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap<float>::calloc(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 256));
    }

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap<float>::calloc<512>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 512));
    }
}
