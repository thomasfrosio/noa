#pragma once

#include "noa/unified/Allocator.h"
#include "noa/unified/Device.h"

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

        [[nodiscard]] constexpr bool dereferencable() const noexcept {
            return m_device.cpu() || m_allocator == Allocator::PINNED ||
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
