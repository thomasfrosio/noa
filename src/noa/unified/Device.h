#pragma once

#include <string_view>

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/utils/Irange.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/string/Parse.hpp"

#include "noa/cpu/Device.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/memory/MemoryPool.h"
#endif

namespace noa {
    struct DeviceMemory { size_t total; size_t free; };

    enum class DeviceType { CPU, GPU };

    /// Unified device/workers. Can either point to a CPU or a GPU.
    class Device {
    public:
        struct DeviceUnchecked {};
    public:
        /// Creates the CPU device.
        Device() = default;

        /// Creates a device.
        /// \param type     CPU or GPU.
        /// \param id       Device ID. This is ignored for the CPU device.
        constexpr explicit Device(DeviceType type, i32 id = 0);

        /// Creates a device.
        /// \param name     Device type and ID. Either "cpu", "gpu", "gpu:N", where N is the ID.
        explicit Device(std::string_view name);

        /// Creates a device, but don't check that the corresponding device exists on the system.
        constexpr explicit Device(DeviceType type, i32 id, DeviceUnchecked);
        explicit Device(std::string_view name, DeviceUnchecked);

    public:
        /// Suspends execution until all previously-scheduled tasks on the specified device have concluded.
        /// On the CPU, the current stream for that device is synchronized. In almost all cases, it is better
        /// to synchronize the streams instead of the devices.
        void synchronize() const;

        /// Explicitly synchronizes, destroys and cleans up all resources associated with the current device in the
        /// current process. The current stream for that device is synchronized and reset to the default stream.
        /// \warning for CUDA devices: It is the caller's responsibility to ensure that resources in all host threads
        ///          (streams and pinned|device arrays) attached to that device are destructed before calling this
        ///          function. The library's internal data will be handled automatically, e.g. FFT plans.
        void reset() const;

        /// Returns a brief printable summary about the device.
        [[nodiscard]] std::string summary() const;

        /// Returns the memory capacity of the device.
        /// If CPU, returns system memory capacity.
        [[nodiscard]] DeviceMemory memory_capacity() const;

        /// Sets the amount of reserved memory in bytes by the device memory pool to hold onto before trying to
        /// release memory back to the OS. Defaults to 0 bytes (i.e. stream synchronization frees the cached memory).
        /// \note This has no effect on the CPU.
        void set_cache_threshold(size_t threshold_bytes) const;

        /// Releases memory back to the OS until the device memory pool contains fewer than \p bytes_to_keep
        /// reserved bytes, or there is no more memory that the allocator can safely release. The allocator cannot
        /// release OS allocations that back outstanding asynchronous allocations.
        /// \note This has no effect on the CPU.
        void trim_cache(size_t bytes_to_keep) const;

        /// Returns the type of device this instance is pointing to.
        [[nodiscard]] constexpr DeviceType type() const noexcept {
            return m_id == -1 ? DeviceType::CPU : DeviceType::GPU;
        }

        /// Whether this device is the CPU.
        [[nodiscard]] constexpr bool is_cpu() const noexcept { return m_id == -1; }

        /// Whether this device is a GPU.
        [[nodiscard]] constexpr bool is_gpu() const noexcept { return m_id != -1; }

        /// Returns the device ID. The ID is always -1 for the CPU.
        /// Otherwise it matches the actual index of the GPU in the system.
        [[nodiscard]] constexpr i32 id() const noexcept { return m_id; }

    public: // Static functions
        /// Gets the current device of the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If \p type is GPU, this function returns the current GPU for the calling host thread.
        ///          If \p type is CPU, this function is not very useful since it simply returns the CPU,
        ///          as would do the default Device constructor.
        [[nodiscard]] static Device current(DeviceType type);

        /// Sets \p device as the current device for the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If this is a GPU, this function set the current GPU for the calling host thread.
        ///          If this is a CPU, this function does nothing
        static void set_current(Device device);

        /// Gets the number of devices of a given type.
        /// Always returns 1 if \p type is CPU.
        [[nodiscard]] static i32 count(DeviceType type);

        /// Whether there's any device available of this type.
        /// Always returns true if \p type is CPU.
        [[nodiscard]] static bool any(DeviceType type);

        /// Gets all devices of a given type.
        /// Always returns a single device if \p type is CPU.
        [[nodiscard]] static std::vector<Device> all(DeviceType type);

        /// Gets the device of this type with the most free memory.
        [[nodiscard]] static Device most_free(DeviceType type);

    private:
        static std::pair<DeviceType, i32> parse_type_and_id_(std::string_view name);
        static void validate_(DeviceType type, i32 id);

    private:
        i32 m_id{-1}; // cpu
    };

    inline bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }

    inline std::ostream& operator<<(std::ostream& os, Device device) {
        if (device.is_cpu())
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
            Device::set_current(*static_cast<Device*>(this));
        }

        ~DeviceGuard() {
            Device::set_current(m_previous_current);
        }

    private:
        Device m_previous_current;
    };

    inline bool operator==(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() == rhs.id(); }
    inline bool operator==(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() == rhs.id(); }

    inline bool operator!=(const Device& lhs, const DeviceGuard& rhs) { return lhs.id() != rhs.id(); }
    inline bool operator!=(const DeviceGuard& lhs, const Device& rhs) { return lhs.id() != rhs.id(); }

    inline std::ostream& operator<<(std::ostream& os, const DeviceGuard& device) {
        if (device.is_cpu())
            return os << "cpu";
        else
            os << "gpu" << ':' << device.id();
        return os;
    }
}

#define NOA_UNIFIED_DEVICE_
#include "noa/unified/Device.inl"
#undef NOA_UNIFIED_DEVICE_
