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

        /// Whether the allocated data can be accessed by the given device type.
        [[nodiscard]] constexpr auto is_reinterpretable(Device::Type type) const noexcept -> bool {
            return device.type() == type or allocator.is_any(
                Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL);
        }

        /// Whether the allocated data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be read/written by the CPU, it does not indicate
        ///       how the library uses the managed data. Indeed, this choice is purely made on the device currently
        ///       associated with the allocated data. For instance, the CPU may access values from a pinned GPU array
        ///       (certain conditions may apply, see Allocator), i.e. it is dereferenceable. However, since the device
        ///       of that array is a GPU, the library will refer to the memory region as GPU-owned and will therefore
        ///       prioritize GPU access.
        [[nodiscard]] constexpr auto is_dereferenceable() const noexcept -> bool {
            return is_reinterpretable(Device::CPU);
        }
    };
}
