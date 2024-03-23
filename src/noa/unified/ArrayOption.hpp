#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/unified/Allocator.hpp"
#include "noa/unified/Device.hpp"

namespace noa::inline types {
    /// Options for Array; simple utility aggregate of a Device and an Allocator.
    class ArrayOption {
    public:
        Device device{};
        Allocator allocator{};

        constexpr ArrayOption& set_device(Device new_device) noexcept {
            device = new_device;
            return *this;
        }

        constexpr ArrayOption& set_allocator(Allocator new_allocator) noexcept {
            allocator = new_allocator;
            return *this;
        }

        /// Whether the allocated data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be read/written on the CPU, it does not indicate
        ///       how the managed data is used. This choice is purely made on the device currently associated with
        ///       the allocated data. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the device is a GPU, the library will refer to the memory region as a
        ///       GPU-own region and will therefore prioritizing GPU access.
        [[nodiscard]] constexpr bool is_dereferenceable() const noexcept {
            return device.is_cpu() or
                   allocator.resource() == MemoryResource::PINNED or
                   allocator.resource() == MemoryResource::MANAGED or
                   allocator.resource() == MemoryResource::MANAGED_GLOBAL;
        }
    };
}
#endif
