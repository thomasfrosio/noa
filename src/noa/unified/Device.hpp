#pragma once

#include <string_view>
#include "noa/core/Error.hpp"
#include "noa/core/utils/Irange.hpp"
#include "noa/core/utils/Strings.hpp"

#include "noa/cpu/Device.hpp"
#ifdef NOA_ENABLE_CUDA
#   include "noa/gpu/cuda/Device.hpp"
#   include "noa/gpu/cuda/MemoryPool.hpp"
#endif

namespace noa::inline types {
    /// Unified device/workers. Can either point to a CPU or a GPU.
    class Device {
    public:
        struct Memory { size_t total; size_t free; };
        // struct Unchecked {};
        enum class Type { CPU, GPU };
        using enum Type;

    public:
        /// Creates the CPU device.
        Device() = default;

        /// Creates a device.
        /// \param type     CPU or GPU.
        /// \param id       Device ID. This is ignored for the CPU device.
        constexpr explicit Device(Type type, i32 id = 0) : m_id(type == CPU ? -1 : id) {
            validate_(type, m_id);
        }

        /// Creates a device from a device name.
        /// Should be either "cpu", "gpu", "gpu:N", where N is the ID, or "gpu:free" to specify
        /// the GPU with the most free memory on the system (at the time of calling this function).
        explicit Device(std::string_view name) {
            m_id = parse_name_and_validate_(name);
        }

        /// Creates a device from a string literal.
        /* implicit */ Device(const char* name) : Device(std::string_view(name)) {}

        /// "Private constructor" to create a device, without checking that the actual device exists on the system.
        constexpr explicit Device(Type type, i32 id, Unchecked) : m_id(type == CPU ? -1 : id) {}

    public:
        /// Suspends execution until all previously scheduled tasks on the specified device have concluded.
        /// In almost all cases, it is better to synchronize the streams instead of the devices.
        /// \warning On the CPU, only the current stream is synchronized.
        void synchronize() const;

        /// Explicitly synchronizes, destroys and cleans up all resources associated with the device in the
        /// current process. The current stream for that device is synchronized and reset to the default stream.
        /// \warning For GPUs, it is the caller's responsibility to ensure that resources in all host threads
        ///          (streams and pinned|device arrays) attached to that device are destructed before calling this
        ///          function. The library's internal data will be handled automatically, e.g. FFT plans.
        void reset() const;

        /// Returns a brief printable summary about the device.
        [[nodiscard]] auto summary() const -> std::string {
            if (is_cpu()) {
                return noa::cpu::Device::summary();
            } else {
                #ifdef NOA_ENABLE_CUDA
                return noa::cuda::Device(this->id(), Unchecked{}).summary();
                #else
                return {};
                #endif
            }
        }

        /// Returns the memory capacity of the device.
        /// If CPU, returns system memory capacity.
        [[nodiscard]] auto memory_capacity() const -> Memory {
            if (is_cpu()) {
                const noa::cpu::DeviceMemory mem_info = noa::cpu::Device::memory();
                return {mem_info.total, mem_info.free};
            } else {
                #ifdef NOA_ENABLE_CUDA
                const auto device = noa::cuda::Device(this->id(), Unchecked{});
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
                #ifdef NOA_ENABLE_CUDA
                const auto device = noa::cuda::Device(id(), Unchecked{});
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
                #ifdef NOA_ENABLE_CUDA
                const auto device = noa::cuda::Device(id(), Unchecked{});
                noa::cuda::MemoryPool(device).trim(bytes_to_keep);
                #else
                (void) bytes_to_keep;
                #endif
            }
        }

        /// Returns the type of device this instance is pointing to.
        [[nodiscard]] constexpr auto type() const noexcept -> Type {
            return m_id == -1 ? CPU : GPU;
        }

        /// Whether this device is the CPU.
        [[nodiscard]] constexpr auto is_cpu() const noexcept -> bool { return m_id == -1; }

        /// Whether this device is a GPU.
        [[nodiscard]] constexpr auto is_gpu() const noexcept -> bool { return m_id != -1; }

        /// Returns the device ID. The ID is always -1 for the CPU.
        /// Otherwise, it matches the actual index of the GPU in the system.
        [[nodiscard]] constexpr auto id() const noexcept -> i32 { return m_id; }

    public: // Static functions
        /// Gets the current device of the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If \p type is GPU, this function returns the current GPU for the calling host thread.
        ///          If \p type is CPU, this function is not very useful since it simply returns the CPU,
        ///          as would do the default Device constructor.
        [[nodiscard]] static auto current(Type type) -> Device {
            if (type == Type::CPU)
                return Device{};
            #ifdef NOA_ENABLE_CUDA
            return Device(Type::GPU, noa::cuda::Device::current().id(), Unchecked{});
            #else
            panic_no_gpu_backend();
            #endif
        }
        [[nodiscard]] static auto current_gpu() -> Device {
            return current(Type::GPU);
        }

        /// Sets \p device as the current device for the calling thread.
        /// \details The underlying state is "thread local", thus thread-safe.
        ///          If this is a GPU, this function set the current GPU for the calling host thread.
        ///          If this is a CPU, this function does nothing
        static void set_current(Device device) {
            if (device.is_gpu()) {
                #ifdef NOA_ENABLE_CUDA
                noa::cuda::Device::set_current(noa::cuda::Device(device.id(), Unchecked{}));
                #else
                panic_no_gpu_backend();
                #endif
            }
        }

        /// Gets the number of devices of a given type.
        /// Always returns 1 if for the CPU.
        [[nodiscard]] static auto count(Type type) -> i32 {
            if (type == CPU) {
                return 1;
            } else {
                #ifdef NOA_ENABLE_CUDA
                return noa::cuda::Device::count();
                #else
                return 0;
                #endif
            }
        }
        [[nodiscard]] static auto count_gpus() -> i32 {
            return count(GPU);
        }

        /// Whether there's any device available of this type.
        /// Always returns true if \p type is CPU.
        [[nodiscard]] static auto is_any(Type type) -> bool {
            return count(type) != 0;
        }

        [[nodiscard]] static auto is_any_gpu() -> bool {
            return count(GPU) != 0;
        }

        /// Gets all devices of a given type.
        /// Always returns a single device if \p type is CPU.
        [[nodiscard]] static auto all(Type type) -> std::vector<Device> {
            if (type == CPU) {
                return {Device(CPU)};
            } else {
                #ifdef NOA_ENABLE_CUDA
                std::vector<Device> devices;
                const i32 count = noa::cuda::Device::count();
                devices.reserve(static_cast<size_t>(count));
                for (auto id: irange(count))
                    devices.emplace_back(GPU, id, Unchecked{});
                return devices;
                #else
                return {};
                #endif
            }
        }

        /// Gets the device of this type with the most free memory.
        [[nodiscard]] static auto most_free(Type type) -> Device {
            if (type == CPU) {
                return Device(CPU);
            } else {
                #ifdef NOA_ENABLE_CUDA
                return Device(GPU, noa::cuda::Device::most_free().id(), Unchecked{});
                #else
                panic("GPU backend is not detected");
                #endif
            }
        }

        /// Gets the device of this type with the most free memory.
        [[nodiscard]] static auto most_free_gpu() -> Device { return most_free(GPU); }

    private:
        static auto parse_name_and_validate_(std::string_view name) -> i32 {
            std::string string = noa::string::to_lower(noa::string::trim(name));
            const size_t length = string.length();

            i32 id{};
            if (noa::string::starts_with(string, "cpu")) {
                check(length == 3, "CPU device name \"{}\" is not supported", string);
                id = -1;

            } else if (noa::string::starts_with(string, "gpu") or noa::string::starts_with(string, "cuda")) {
                const size_t offset = string[0] == 'c' ? 1 : 0;
                if (length == 3 + offset) {
                    id = 0;
                    check(is_any_gpu(), "GPU device ID 0 is not valid");
                } else if (length >= 5 + offset and string[3 + offset] == ':') {
                    std::string_view specifier = noa::string::offset_by(string, 4 + offset);
                    std::optional<u32> result = noa::string::parse<u32>(specifier);
                    if (result.has_value()) {
                        id = static_cast<i32>(*result);
                        check(id < count_gpus(), "GPU device ID {} is not valid", id);
                    } else if (specifier == "free") {
                        id = most_free_gpu().id();
                    } else {
                        panic("Failed to parse the GPU ID, name or specifier: \"{}\"", string);
                    }
                } else {
                    panic("GPU device name \"{}\" is not supported", string);
                }

            } else {
                panic("\"{}\" is not a valid device name", string);
            }
            return id;
        }

        static constexpr void validate_(Type type, i32 id) {
            switch (type) {
                case CPU: {
                    check(id == -1, "The device ID for the CPU should be -1, but got {}", id);
                    break;
                }
                case GPU: {
                    check(id >= 0, "GPU device ID should be positive, but got {}", id);
                    check(id < count_gpus(), "GPU device ID {} is not valid", id);
                }
            }
        }

    private:
        i32 m_id{-1}; // defaults to cpu
    };

    inline bool operator==(const Device& lhs, const Device& rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(const Device& lhs, const Device& rhs) { return lhs.id() != rhs.id(); }

    inline auto operator<<(std::ostream& os, const Device& device) -> std::ostream& {
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
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::Device> : ostream_formatter {};
    template<> struct formatter<noa::DeviceGuard> : ostream_formatter {};
}
