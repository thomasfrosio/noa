#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <string_view>
#include "noa/core/Exception.hpp"
#include "noa/core/utils/Irange.hpp"
#include "noa/core/utils/Strings.hpp"

#include "noa/cpu/Device.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Device.hpp"
#include "noa/gpu/cuda/MemoryPool.hpp"
#endif

namespace noa::inline types {
    /// Unified device/workers. Can either point to a CPU or a GPU.
    class Device {
    public:
        struct Memory { size_t total; size_t free; };
        struct Unchecked {};
        enum class Type { CPU, GPU };
        using enum Type;

    public:
        /// Creates the CPU device.
        Device() = default;

        /// Creates a device.
        /// \param type     CPU or GPU.
        /// \param id       Device ID. This is ignored for the CPU device.
        constexpr explicit Device(Type type, i32 id = 0) : m_id(type == Type::CPU ? -1 : id) {
            validate_(type, m_id);
        }

        /// Creates a device.
        /// \param name     Device type and ID. Either "cpu", "gpu", "gpu:N", where N is the ID.
        explicit Device(std::string_view name) {
            const auto [type, id] = parse_type_and_id_(name);
            validate_(type, id);
            m_id = id;
        }

        /// Creates a device from a string literal.
        /* implicit */ Device(const char* name) : Device(std::string_view(name)) {}

        /// "Private constructor" to creates a device, without checking that the actual device exists on the system.
        constexpr explicit Device(Type type, i32 id, Unchecked) : m_id(type == Type::CPU ? -1 : id) {}

    public:
        /// Suspends execution until all previously-scheduled tasks on the specified device have concluded.
        /// On the CPU, the current stream for that device is synchronized. In almost all cases, it is better
        /// to synchronize the streams instead of the devices.
        void synchronize() const;

        /// Explicitly synchronizes, destroys and cleans up all resources associated with the current device in the
        /// current process. The current stream for that device is synchronized and reset to the default stream.
        /// \warning for CUDA devices, it is the caller's responsibility to ensure that resources in all host threads
        ///          (streams and pinned|device arrays) attached to that device are destructed before calling this
        ///          function. The library's internal data will be handled automatically, e.g. FFT plans.
        void reset() const;

        /// Returns a brief printable summary about the device.
        [[nodiscard]] std::string summary() const {
            if (is_cpu()) {
                return noa::cpu::Device::summary();
            } else {
                #ifdef NOA_ENABLE_CUDA
                return noa::cuda::Device(this->id(), noa::cuda::Device::DeviceUnchecked{}).summary();
                #else
                return {};
                #endif
            }
        }

        /// Returns the memory capacity of the device.
        /// If CPU, returns system memory capacity.
        [[nodiscard]] Memory memory_capacity() const {
            if (is_cpu()) {
                const noa::cpu::DeviceMemory mem_info = noa::cpu::Device::memory();
                return {mem_info.total, mem_info.free};
            } else {
                #ifdef NOA_ENABLE_CUDA
                const auto device = noa::cuda::Device(this->id(), noa::cuda::Device::DeviceUnchecked{});
                const auto mem_info = device.memory();
                return {mem_info.total, mem_info.free};
                #else
                return {};
                #endif
            }
        }

        /// Sets the amount of reserved memory in bytes by the device memory pool to hold onto before trying to
        /// release memory back to the OS. Defaults to 0 bytes (i.e. stream synchronization frees the cached memory).
        /// \note This has no effect on the CPU.
        void set_cache_threshold(size_t threshold_bytes) const {
            if (is_gpu()) {
                #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
                const auto device = noa::cuda::Device(id(), noa::cuda::Device::DeviceUnchecked{});
                noa::cuda::MemoryPool(device).set_threshold(threshold_bytes);
                #else
                (void) threshold_bytes;
                #endif
            }
        }

        /// Releases memory back to the OS until the device memory pool contains fewer than \p bytes_to_keep
        /// reserved bytes, or there is no more memory that the allocator can safely release. The allocator cannot
        /// release OS allocations that back outstanding asynchronous allocations.
        /// \note This has no effect on the CPU.
        void trim_cache(size_t bytes_to_keep) const {
            if (is_gpu()) {
                #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
                const auto device = noa::cuda::Device(id(), noa::cuda::Device::DeviceUnchecked{});
                noa::cuda::MemoryPool(device).trim(bytes_to_keep);
                #else
                (void) bytes_to_keep;
                #endif
            }
        }

        /// Returns the type of device this instance is pointing to.
        [[nodiscard]] constexpr Type type() const noexcept {
            return m_id == -1 ? Type::CPU : Type::GPU;
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
        [[nodiscard]] static Device current(Type type) {
            if (type == Type::CPU)
                return Device{};
            #ifdef NOA_ENABLE_CUDA
            return Device(Type::GPU, noa::cuda::Device::current().id(), Unchecked{});
            #else
            panic("No GPU backend detected");
            #endif
        }
        [[nodiscard]] static Device current_gpu() {
            return current(Type::GPU);
        }

        /// Sets \p device as the current device for the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If this is a GPU, this function set the current GPU for the calling host thread.
        ///          If this is a CPU, this function does nothing
        static void set_current(Device device) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                noa::cuda::Device::set_current(noa::cuda::Device(device.id(), noa::cuda::Device::DeviceUnchecked{}));
                #else
                panic("No GPU backend detected");
                #endif
            }
        }

        /// Gets the number of devices of a given type.
        /// Always returns 1 if \p type is CPU.
        [[nodiscard]] static i32 count(Type type) {
            if (type == Type::CPU) {
                return 1;
            } else {
                #ifdef NOA_ENABLE_CUDA
                return noa::cuda::Device::count();
                #else
                return 0;
                #endif
            }
        }
        [[nodiscard]] static i32 count_gpus() {
            return count(Type::GPU);
        }

        /// Whether there's any device available of this type.
        /// Always returns true if \p type is CPU.
        [[nodiscard]] static bool is_any(Type type) {
            return count(type) != 0;
        }

        [[nodiscard]] static bool is_any_gpu() {
            return count(Type::GPU) != 0;
        }

        /// Gets all devices of a given type.
        /// Always returns a single device if \p type is CPU.
        [[nodiscard]] static std::vector<Device> all(Type type) {
            if (type == Type::CPU) {
                return {Device(Type::CPU)};
            } else {
                #ifdef NOA_ENABLE_CUDA
                std::vector<Device> devices;
                const i32 count = noa::cuda::Device::count();
                devices.reserve(static_cast<size_t>(count));
                for (auto id: irange(count))
                    devices.emplace_back(Type::GPU, id, Unchecked{});
                return devices;
                #else
                return {};
                #endif
            }
        }

        /// Gets the device of this type with the most free memory.
        [[nodiscard]] static Device most_free(Type type) {
            if (type == Type::CPU) {
                return Device(Type::CPU);
            } else {
                #ifdef NOA_ENABLE_CUDA
                return Device(Type::GPU, noa::cuda::Device::most_free().id(), Unchecked{});
                #else
                panic("GPU backend is not detected");
                #endif
            }
        }

    private:
        static std::pair<Type, i32> parse_type_and_id_(std::string_view name) {
            std::string string = ns::to_lower(ns::trim(name));
            const size_t length = string.length();

            i32 id{};
            Type type{};
            if (ns::starts_with(string, "cpu")) {
                if (length == 3) {
                    id = -1;
                    type = Type::CPU;
                } else {
                    panic("CPU device name \"{}\" is not supported", string);
                }

            } else if (ns::starts_with(string, "gpu")) {
                if (length == 3) {
                    id = 0;
                } else if (length >= 5 and string[3] == ':') {
                    auto result = ns::parse<i32>(ns::offset_by(string, 4));
                    check(result, "Failed to parse the device ID: \"{}\"", string);
                    id = *result;
                } else {
                    panic("GPU device name \"{}\" is not supported", string);
                }
                type = Type::GPU;

            } else if (ns::starts_with(string, "cuda")) {
                if (length == 4) {
                    id = 0;
                } else if (length >= 6 and string[4] == ':') {
                    auto result = ns::parse<i32>(ns::offset_by(string, 5));
                    check(result, "Failed to parse the device ID: \"{}\"", string);
                    id = *result;
                } else {
                    panic("CUDA device name \"{}\" is not supported", string);
                }
                type = Type::GPU;
            } else {
                panic("\"{}\" is not a valid device name", string);
            }
            return {type, id};
        }

        static constexpr void validate_(Type type, i32 id) {
            switch (type) {
                case Type::CPU: {
                    if (id != -1)
                        panic("The device ID for the CPU should be -1, but got {}", id);
                    break;
                }
                case Type::GPU: {
                    if (id < 0)
                        panic("GPU device ID should be positive, but got {}", id);

                    #ifdef NOA_ENABLE_CUDA
                    const i32 count = noa::cuda::Device::count();
                    if (id + 1 > count)
                        panic("CUDA device ID \"{}\" does not match any of CUDA device(s) detected (count={})", id, count);
                    break;
                    #else
                    panic("GPU backend is not detected");
                    #endif
                }
            }
        }

    private:
        i32 m_id{-1}; // defaults to cpu
    };

    inline bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }

    inline std::ostream& operator<<(std::ostream& os, Device device) {
        if (device.is_cpu())
            return os << "cpu";
        return os << "gpu" << ':' << device.id();
    }

    /// A device that sets itself as the current device for the remainder of the scope.
    /// \note CPU guards have no effect since they simply refer to the CPU and CPU devices
    ///       have no state (there's only one CPU). Really, this is only useful to switch
    ///       between GPUs in a non-destructive way.
    class DeviceGuard : public Device {
    public:
        template<typename ... Args>
        explicit DeviceGuard(Args&& ...args) :
            Device(std::forward<Args>(args)...),
            m_previous_current(current(this->type()))
        {
            set_current(*static_cast<Device*>(this));
        }

        ~DeviceGuard() {
            set_current(m_previous_current);
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
        return os << "gpu" << ':' << device.id();
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::Device> : ostream_formatter {};
    template<> struct formatter<noa::DeviceGuard> : ostream_formatter {};
}

#endif
