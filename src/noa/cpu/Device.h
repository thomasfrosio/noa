#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu {
    struct DeviceMemory { size_t free; size_t total; }; // bytes
    struct DeviceCore { size_t logical; size_t physical; };
    struct DeviceCache { size_t size; size_t line_size; }; // bytes

    class Device {
    public:
        // Retrieves the system memory usage, in bytes.
        [[nodiscard]] static DeviceMemory memory();

        // Retrieves the amount of logical (hyper-threads) and physical cores.
        [[nodiscard]] static DeviceCore cores();

        // Retrieves the cache size and line count for a given cache level.
        [[nodiscard]] static DeviceCache cache(int level);

        // Retrieves the processor printable name.
        [[nodiscard]] static std::string name();

        // Retrieves a printable summary about the CPU and system memory.
        [[nodiscard]] static std::string summary();

        // Clears the internal data of the CPU backend.
        static void reset();
    };
}
