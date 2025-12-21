#pragma once

#include <cstddef>
#include <string>
#include "noa/base/Traits.hpp"

namespace noa::cpu {
    struct DeviceMemory { usize free; usize total; }; // bytes
    struct DeviceCore { usize logical; usize physical; };
    struct DeviceCache { usize size; usize line_size; }; // bytes

    class Device {
    public:
        /// The library keeps track of some global resources that need to be deleted upon Device::reset().
        /// This is the mechanism by which we can attach resources to a device: callbacks can be added (and removed)
        /// to an internal thread-safe set managed by Device. When calling Device::reset(), these callbacks are
        /// called once after the device synchronization but before the reset.
        using reset_callback_type = void (*)(Device);
        static void add_reset_callback(reset_callback_type);
        static void remove_reset_callback(reset_callback_type);

        /// Retrieves the system memory usage, in bytes.
        [[nodiscard]] static auto memory() -> DeviceMemory;

        /// Retrieves the number of logical (hyper-threads) and physical cores.
        [[nodiscard]] static auto cores() -> DeviceCore;

        /// Retrieves the cache size and line count for a given cache level.
        [[nodiscard]] static auto cache(int level) -> DeviceCache;

        /// Retrieves the processor printable name.
        [[nodiscard]] static auto name() -> std::string;

        /// Retrieves a printable summary about the CPU and system memory.
        [[nodiscard]] static auto summary() -> std::string;

    public:
        constexpr Device() = default;

        /// Clears the internal data of the CPU backend.
        void reset() const;
    };
}
