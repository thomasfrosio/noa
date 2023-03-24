#pragma once

#include "noa/unified/Allocator.hpp"
#include "noa/unified/Device.hpp"

namespace noa {
    /// Options for Array(s).
    class ArrayOption {
    public: // Constructors
        /// Sets the array options.
        /// \param device       Device of the array. Defaults to the CPU.
        /// \param allocator    Allocator of the array. Defaults to the default device allocator.
        constexpr /*implicit*/ ArrayOption(Device device, Allocator allocator)
                : m_device(device), m_allocator(allocator) {}

        constexpr ArrayOption() = default;
        constexpr /*implicit*/ ArrayOption(Device device) : m_device(device) {}
        constexpr /*implicit*/ ArrayOption(Allocator allocator) : m_allocator(allocator) {}

    public: // Setters
        constexpr ArrayOption& device(Device device) noexcept {
            m_device = device;
            return *this;
        }

        constexpr ArrayOption& allocator(Allocator allocator) noexcept {
            m_allocator = allocator;
            return *this;
        }

        /// Whether the allocated data can be accessed by CPU threads.
        /// \note While it indicates whether the managed data can be read/written on the CPU, it does not indicate
        ///       how the managed data is used. This choice is purely made on the device currently associated with
        ///       the allocated data. For instance, pinned memory can be dereferenced by the CPU, so this function will
        ///       returned true, but if the device is a GPU, the library will refer to the memory region as a
        ///       GPU-own region and will therefore prioritizing GPU access.
        [[nodiscard]] constexpr bool is_dereferenceable() const noexcept {
            return m_device.is_cpu() || m_allocator == Allocator::PINNED ||
                   m_allocator == Allocator::MANAGED || m_allocator == Allocator::MANAGED_GLOBAL;
        }

    public: // Getters
        [[nodiscard]] constexpr Device device() const noexcept { return m_device; }
        [[nodiscard]] constexpr Allocator allocator() const noexcept { return m_allocator; }

    private:
        Device m_device{};
        Allocator m_allocator{Allocator::DEFAULT};
        // TODO Switch to uint32_t bitset?
    };
}
