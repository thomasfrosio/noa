#pragma once

#include <string_view>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/utils/Irange.h"
#include "noa/common/string/Format.h"
#include "noa/common/string/Parse.h"

#include "noa/cpu/Device.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/memory/MemoryPool.h"
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
        /// Creates the CPU device.
        Device() = default;

        /// Creates a device.
        /// \param type     Whether it is CPU or a GPU.
        /// \param id       Device ID. This is ignored for CPU devices.
        /// \param unsafe   Whether the device should checks that the corresponding device
        ///                 exists on the system. This is ignored for CPU devices.
        constexpr explicit Device(Type type, int id = 0, bool unsafe = false);

        /// Creates a device.
        /// \param name     Device type and ID. Either "cpu", "gpu", "gpu:N", where N is the ID.
        /// \param unsafe   Whether the device should checks that the corresponding device
        ///                 exists on the system. This is ignored for CPU devices.
        explicit Device(std::string_view name, bool unsafe = false);

    public:
        /// Suspends execution until all previously-scheduled tasks on the specified device have concluded.
        /// On the CPU, the current stream for that device is synchronized. In almost all cases, it is better
        /// to synchronize the streams instead of the devices.
        void synchronize() const;

        /// Explicitly synchronizes, destroys and cleans up all resources associated with the current device in the
        /// current process. The current stream for that device is synchronized and reset to the default stream.
        /// \warning for CUDA devices: It is the caller's responsibility to ensure that resources in all host threads
        ///          (streams and pinned|device arrays) attached to that device are destructed before calling this
        ///          function. The library's internal data will be handled automatically, e.g. FFT plans or textures.
        void reset() const;

        /// Returns a brief printable summary about the device.
        [[nodiscard]] std::string summary() const;

        /// Returns the memory capacity of the device.
        /// If CPU, returns system memory capacity.
        [[nodiscard]] DeviceMemory memory() const;

        /// Sets the amount of reserved memory in bytes by the device memory pool to hold onto before trying to
        /// release memory back to the OS. Defaults to 0 bytes (i.e. stream synchronization frees the cached memory).
        /// \note This has no effect on the CPU.
        void memoryThreshold(size_t threshold_bytes) const;

        /// Releases memory back to the OS until the device memory pool contains fewer than \p bytes_to_keep
        /// reserved bytes, or there is no more memory that the allocator can safely release. The allocator cannot
        /// release OS allocations that back outstanding asynchronous allocations.
        /// \note This has no effect on the CPU.
        void memoryTrim(size_t bytes_to_keep) const;

        /// Returns the type of device this instance is pointing to.
        [[nodiscard]] constexpr Type type() const noexcept {
            return m_id == -1 ? Type::CPU : Type::GPU;
        }

        /// Whether this device is the CPU.
        [[nodiscard]] constexpr bool cpu() const noexcept { return m_id == -1; }

        /// Whether this device is a GPU.
        [[nodiscard]] constexpr bool gpu() const noexcept { return m_id != -1; }

        /// Returns the device ID. The ID is always -1 for the CPU.
        /// Otherwise it matches the actual index of the GPU in the system.
        [[nodiscard]] constexpr int id() const noexcept { return m_id; }

    public: // Static functions
        /// Gets the current device of the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If \p type is GPU, this function returns the current GPU for the calling host thread.
        ///          If \p type is CPU, this function is not very useful since it simply returns the CPU,
        ///          as would do the default Device constructor.
        static Device current(Type type);

        /// Sets \p device as the current device for the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If this is a GPU, this function set the current GPU for the calling host thread.
        ///          If this is a CPU, this function does nothing
        static void current(Device device);

        /// Gets the number of devices of a given type.
        /// Always returns 1 if \p type is CPU.
        static size_t count(Type type);

        /// Whether there's any device available of this type.
        /// Always returns true if \p type is CPU.
        static bool any(Type type);

        /// Gets all devices of a given type.
        /// Always returns a single device if \p type is CPU.
        static std::vector<Device> all(Type type);

        /// Gets the device of this type with the most free memory.
        static Device mostFree(Type type);

    private:
        static int parse_(std::string_view str);
        static void validate_(int id);

    private:
        int m_id{-1}; // cpu
    };

    inline bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }

    inline std::ostream& operator<<(std::ostream& os, Device device) {
        if (device.cpu())
            return os << "cpu";
        else
            os << "gpu" << ':' << device.id();
        return os;
    }
}

namespace noa {
    /// A device that sets itself as the current device for the remainder of the scope.
    /// \note CPU guards have no effect since they simply refer to the CPU and CPU devices
    ///       have no state (there's only one CPU). Really, this is only useful to switch
    ///       between GPUs in a non-destructive way.
    class DeviceGuard : public Device {
    public:
        template<typename ... Args>
        explicit DeviceGuard(Args&& ...args)
                : Device(std::forward<Args>(args)...),
                  m_previous_current(Device::current(this->type())) {
            Device::current(*static_cast<Device*>(this));
        }

        ~DeviceGuard() {
            Device::current(m_previous_current);
        }

    private:
        Device m_previous_current;
    };

    inline bool operator==(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() == rhs.id(); }
    inline bool operator==(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() == rhs.id(); }

    inline bool operator!=(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() != rhs.id(); }
    inline bool operator!=(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() != rhs.id(); }

    inline std::ostream& operator<<(std::ostream& os, const DeviceGuard& device) {
        if (device.cpu())
            return os << "cpu";
        else
            os << "gpu" << ':' << device.id();
        return os;
    }
}

#define NOA_UNIFIED_DEVICE_
#include "noa/unified/Device.inl"
#undef NOA_UNIFIED_DEVICE_
