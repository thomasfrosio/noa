#pragma once

#include <string_view>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Irange.h"
#include "noa/common/string/Format.h"
#include "noa/common/string/Parse.h"

#include "noa/cpu/Device.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Device.h"
#endif

namespace noa {
    struct DeviceMemory { size_t total; size_t free; };

    /// Unified device/workers. Can either point to a CPU or a GPU.
    class Device {
    public:
        enum Type {
            CPU, GPU
        };

    public:
        /// Creates the current device.
        Device();

        /// Creates a device based on a type and ID.
        constexpr explicit Device(Type type, int id = 0, bool unsafe = false);

        /// Creates a device based on a name (either "cpu", "gpu", "gpu:N", "cuda" or "cuda:N", where N is the ID.
        explicit Device(std::string_view device, bool unsafe = false);

    public:
        /// Suspends execution until all previously-scheduled tasks on the specified device have concluded.
        void synchronize() const;

        /// Explicitly destroys and cleans up all resources associated
        /// with the current device in the current process. If CPU, do nothing.
        void reset() const;

        /// Returns a brief printable summary about the device.
        [[nodiscard]] std::string summary() const;

        /// Returns the memory capacity of the device.
        /// If CPU, returns system memory capacity.
        [[nodiscard]] DeviceMemory memory() const;

        [[nodiscard]] constexpr Type type() const noexcept { return m_id == -1 ? Type::CPU : Type::GPU; }
        [[nodiscard]] constexpr bool cpu() const noexcept { return m_id == -1; }
        [[nodiscard]] constexpr bool gpu() const noexcept { return m_id != -1; }
        [[nodiscard]] constexpr int id() const noexcept { return m_id; }

    public: // Static functions
        /// Gets the current device of the calling thread. Defaults to the CPU.
        /// The underlying state is "thread local", thus thread-safe.
        static Device current();

        /// Sets \p device as the current device for the calling thread.
        /// The underlying state is "thread local", thus thread-safe.
        static void current(Device device);

        /// Gets the number of devices of a given type.
        static size_t count(Type type);

        /// Whether there's any device available of this type.
        static bool any(Type type);

        /// Gets all devices of a given type.
        static std::vector<Device> all(Type type);

        /// Gets the device of this type with the most free memory.
        static Device mostFree(Type type);

    private:
        static int parse_(std::string_view str);
        static void validate_(int id);

    private:
        int m_id{-1}; // cpu
    };

    NOA_IH bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }
}

namespace noa {
    /// A device that sets itself as the current device for the remainder of the scope.
    class DeviceGuard : public Device {
    public:
        template<typename ... Args>
        explicit DeviceGuard(Args&& ...args)
                : Device(std::forward<Args>(args)...),
                  m_previous_current(Device::current()) {
            Device::current(*static_cast<Device*>(this));
        }

        ~DeviceGuard() {
            Device::current(m_previous_current);
        }

    private:
        Device m_previous_current;
    };

    NOA_IH bool operator==(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator==(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() == rhs.id(); }

    NOA_IH bool operator!=(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() != rhs.id(); }
    NOA_IH bool operator!=(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() != rhs.id(); }
}

#define NOA_UNIFIED_DEVICE_
#include "noa/unified/Device.inl"
#undef NOA_UNIFIED_DEVICE_
