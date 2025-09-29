#include <noa/cpu/Allocators.hpp>

#include "Catch.hpp"

TEST_CASE("cpu::AllocatorHeap") {
    size_t allocated_start = noa::cpu::AllocatorHeap::bytes_currently_allocated();

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap::allocate<float>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 256));
    }
    REQUIRE(allocated_start == noa::cpu::AllocatorHeap::bytes_currently_allocated());

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap::allocate<float, 512>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 512));
    }
    REQUIRE(allocated_start == noa::cpu::AllocatorHeap::bytes_currently_allocated());

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap::calloc<float>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 256));
    }
    REQUIRE(allocated_start == noa::cpu::AllocatorHeap::bytes_currently_allocated());

    for (int i = 1; i < 12; i++) {
        std::unique_ptr buffer = noa::cpu::AllocatorHeap::calloc<float, 512>(i);
        REQUIRE(buffer != nullptr);
        REQUIRE(!(reinterpret_cast<std::uintptr_t>(buffer.get()) % 512));
    }
    REQUIRE(allocated_start == noa::cpu::AllocatorHeap::bytes_currently_allocated());
}
