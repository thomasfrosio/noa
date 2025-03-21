#pragma once

#include <cstddef>
#include <string>

namespace noa::cpu {
    struct DeviceMemory { size_t free; size_t total; }; // bytes
    struct DeviceCore { size_t logical; size_t physical; };
    struct DeviceCache { size_t size; size_t line_size; }; // bytes

    class Device {
    public:
        // Retrieves the system memory usage, in bytes.
        [[nodiscard]] static auto memory() -> DeviceMemory;

        // Retrieves the number of logical (hyper-threads) and physical cores.
        [[nodiscard]] static auto cores() -> DeviceCore;

        // Retrieves the cache size and line count for a given cache level.
        [[nodiscard]] static auto cache(int level) -> DeviceCache;

        // Retrieves the processor printable name.
        [[nodiscard]] static auto name() -> std::string;

        // Retrieves a printable summary about the CPU and system memory.
        [[nodiscard]] static auto summary() -> std::string;

        // Clears the internal data of the CPU backend.
        static void reset();
    };
}
