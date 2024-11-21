#pragma once

#include "noa/unified/Allocator.hpp"
#include "noa/unified/Device.hpp"

namespace noa::inline types {
    /// Options for Array; simple utility aggregate of a Device and an Allocator.
    class ArrayOption {
    public:
        Device device{};
        Allocator allocator{};

        constexpr auto set_device(Device new_device) noexcept -> ArrayOption& {
            device = new_device;
            return *this;
        }

        constexpr auto set_allocator(Allocator new_allocator) noexcept -> ArrayOption& {
            allocator = new_allocator;
            return *this;
        }

        /// Whether the allocated data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be read/written on the CPU, it does not indicate
        ///       how the managed data is used. This choice is purely made on the device currently associated with
        ///       the allocated data. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the device is a GPU, the library will refer to the memory region as a
        ///       GPU-own region and will therefore prioritize GPU access.
        [[nodiscard]] constexpr auto is_dereferenceable() const noexcept -> bool {
            return device.is_cpu() or allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL);
        }
    };
}
