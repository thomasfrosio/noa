#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "noa/base/Strings.hpp"
#include "noa/runtime/cuda/Runtime.hpp"
#include "noa/runtime/cuda/Error.hpp"

namespace noa::gpu {
#if defined(NOA_ENABLE_CUDA)
    using DeviceProperties = cudaDeviceProp;
    using DeviceAttribute  = cudaDeviceAttr;
    using DeviceLimit      = cudaLimit;
#elif defined(NOA_ENABLE_HIP)
    using DeviceProperties = hipDeviceProp_t;
    using DeviceAttribute  = hipDeviceAttribute_t;
    using DeviceLimit      = hipLimit_t;
#elif defined(NOA_ENABLE_SYCL)
    using DeviceLimit = void*; // not supported
    using DeviceProperties = sycl::device;  // no properties struct, queries go through get_info<>()
    namespace details {
        auto sycl_devices() -> const std::vector<sycl::device>&;
        auto sycl_current_device_id() -> i32;
        void sycl_set_current_device_id(i32 id);
        auto sycl_device_context(i32 id) -> const sycl::context&;
    }
#endif

    struct DeviceMemory { usize free; usize total; }; // bytes
    struct DeviceCapability { int major; int minor; };

    /// A GPU device.
    class Device {
    public:
        using device_callback_type = void (*)(i32);
        static void add_synchronize_callback(device_callback_type);
        static void remove_synchronize_callback(device_callback_type);
        static void add_reset_callback(device_callback_type);
        static void remove_reset_callback(device_callback_type);

    public:
        /// Creates the current GPU device.
        Device() : m_id(current().m_id) {}

        /// Creates a GPU device from an ID within range [0, Device::count()).
        constexpr explicit Device(std::integral auto id) : m_id(static_cast<i32>(id)) {
            validate_(m_id);
        }

        constexpr explicit Device(std::integral auto id, Unchecked) : m_id(static_cast<i32>(id)) {}

    public:
        /// Suspends execution until all previously scheduled tasks on the specified device have concluded.
        /// Before calling the backend synchronization function, all registered synchronization callbacks are called.
        ///
        /// \note (CUDA) Depending on the host synchronization scheduling policy set for the device,
        ///       the calling thread will either yield, spin or block until this completion.
        void synchronize() const;

        /// Explicitly synchronizes, destroys and cleans up all resources associated with the current device in the
        /// current process. Any subsequent API call to this device will reinitialize the device.
        /// Before calling the backend reset function, all registered reset callbacks are called.
        ///
        /// It is the caller's responsibility to ensure that resources (e.g. streams, pinned|device|CUDA arrays,
        /// and CUDA textures) attached to that device are destructed before calling this function.
        /// The library's internal data are handled automatically using reset callbacks.
        void reset() const;

        /// Retrieves the properties of the device.
        [[nodiscard]] auto properties() const -> DeviceProperties {
#if defined(NOA_ENABLE_CUDA)
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, m_id));
            return properties;
#elif defined(NOA_ENABLE_HIP)
            hipDeviceProp_t properties{};
            check(hipGetDeviceProperties(&properties, m_id));
            return properties;
#elif defined(NOA_ENABLE_SYCL)
            return details::sycl_devices()[static_cast<usize>(m_id)]; // reference-counted sycl::device
#endif
        }

#if defined(NOA_ENABLE_CUDA)
        /// Retrieves an attribute (one of the many that composes a device_prop_t) from a device.
        /// The following attributes require PCIe reads and are therefore much slower to get:
        /// cudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate,
        /// and cudaDevAttrSingleToDoublePrecisionPerfRatio.
        [[nodiscard]] auto attribute(cudaDeviceAttr attribute) const -> int {
            int attribute_value;
            check(cudaDeviceGetAttribute(&attribute_value, attribute, m_id));
            return attribute_value;
        }
#elif defined(NOA_ENABLE_HIP)
        [[nodiscard]] auto attribute(hipDeviceAttribute_t attribute) const -> int {
            int attribute_value;
            check(hipDeviceGetAttribute(&attribute_value, attribute, m_id));
            return attribute_value;
        }
#elif defined(NOA_ENABLE_SYCL)
        /// Retrieves an attribute from a device.
        /// In SYCL, attributes are compile-time descriptors,
        /// e.g.: device.attribute<sycl::info::device::max_work_group_size>().
        template<typename Descriptor>
        [[nodiscard]] auto attribute() const -> Descriptor::return_type {
            return details::sycl_devices()[static_cast<usize>(m_id)].get_info<Descriptor>();
        }
#endif

        /// Gets the device's hardware architecture generation numeric designator.
        [[nodiscard]] auto architecture() const -> i32 {
#if defined(NOA_ENABLE_CUDA)
            i32 major;
            check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            return major;
#elif defined(NOA_ENABLE_HIP)
            // On AMD this is the gfx major (e.g. 9 for gfx90a/gfx942, 10 for RDNA1/2, 11 for RDNA3),
            // not an NVIDIA-style generation. For the exact target, use hipDeviceProp_t::gcnArchName.
            // TODO check this
            i32 major;
            check(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, m_id));
            return major;
#elif defined(NOA_ENABLE_SYCL)
            // TODO No equivalent. Look at DPC++ and AdaptiveCpp equivalent.
            return 0;
#endif
        }

        /// Gets the device's compute capability (major, minor) numeric designator.
        [[nodiscard]] auto capability() const -> DeviceCapability {
#if defined(NOA_ENABLE_CUDA)
            int major, minor;
            check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, m_id));
            return {major, minor};
#elif defined(NOA_ENABLE_HIP)
            // gfx90a -> {9, 0}, gfx942 -> {9, 4}, gfx1100 -> {11, 0}. Same caveat as above.
            int major, minor;
            check(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, m_id));
            check(hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, m_id));
            return {major, minor};
#elif defined(NOA_ENABLE_SYCL)
            // TODO No equivalent. Look at DPC++ and AdaptiveCpp equivalent.
            return {};
#endif
        }

        /// Returns the free and total amount of memory available for allocation by the device, in bytes.
        [[nodiscard]] auto memory() const -> DeviceMemory {
#if defined(NOA_ENABLE_CUDA)
            usize mem_free, mem_total;
            const Device previous_current = Device::current();
            set_current(*this);
            check(cudaMemGetInfo(&mem_free, &mem_total));
            set_current(previous_current);
            return {mem_free, mem_total};
#elif defined(NOA_ENABLE_HIP)
            usize mem_free, mem_total;
            const Device previous_current = Device::current();
            set_current(*this);
            check(hipMemGetInfo(&mem_free, &mem_total));
            set_current(previous_current);
            return {mem_free, mem_total};
#elif defined(NOA_ENABLE_SYCL)
            const sycl::device& device = details::sycl_devices()[static_cast<usize>(m_id)];
            const auto mem_total = device.get_info<sycl::info::device::global_mem_size>();
            usize mem_free = mem_total; // TODO free memory is not part of SYCL 2020
#   if defined(__SYCL_COMPILER_VERSION) // DPC++ only
            if (device.has(sycl::aspect::ext_intel_free_memory))
                mem_free = device.get_info<sycl::ext::intel::info::device::free_memory>();
#   elif defined(__ACPP__)
            // See https://adaptivecpp.github.io/AdaptiveCpp/extensions/#ACPP_EXT_DEVICE_FREE_MEMORY
            if (device.has(sycl::aspect::AdaptiveCpp_free_memory)) // implemented for CUDA and HIP
                mem_free = device.get_info<sycl::info::device::AdaptiveCpp_free_memory>();
#   endif
            return {mem_free, mem_total};
#endif
        }

        /// Retrieves a human-readable string of the GPU runtime version.
        [[nodiscard]] auto runtime_version() const -> std::string {
#if defined(NOA_ENABLE_CUDA)
            i32 version;
            check(cudaRuntimeGetVersion(&version));
            const int major = version / 1000;
            const int minor = version / 10 - major * 100;
            return fmt::format("{}.{}", major, minor);
#elif defined(NOA_ENABLE_HIP)
            i32 version;
            check(hipRuntimeGetVersion(&version));
            return fmt::format("{}.{}", version / 10'000'000, (version / 100'000) % 100);
#elif defined(NOA_ENABLE_SYCL)
            return details::sycl_devices()[static_cast<usize>(m_id)].get_info<sycl::info::device::version>();
#endif
        }

        /// Retrieves a human-readable string of the GPU driver version.
        [[nodiscard]] auto driver_version() const -> std::string {
#if defined(NOA_ENABLE_CUDA)
            i32 version;
            check(cudaDriverGetVersion(&version));
            const int major = version / 1000;
            const int minor = version / 10 - major * 100;
            return fmt::format("{}.{}", major, minor);
#elif defined(NOA_ENABLE_HIP)
            i32 version;
            check(hipDriverGetVersion(&version));
            return fmt::format("{}.{}", version / 10'000'000, (version / 100'000) % 100);
#elif defined(NOA_ENABLE_SYCL)
            return details::sycl_devices()[static_cast<usize>(m_id)].get_info<sycl::info::device::driver_version>();
#endif
        }

        /// Retrieves a summary of the device.
        [[nodiscard]] auto summary() const -> std::string {
            const DeviceProperties prop = properties();
            const DeviceMemory mem = memory();

#if defined(NOA_ENABLE_CUDA)
            const char* backend_name = "cuda";
            std::string name = prop.name;
            std::string capability_str = fmt::format("{}.{}", prop.major, prop.minor);
#elif defined(NOA_ENABLE_HIP)
            const char* backend_name = "hip";
            std::string name = prop.name;
            std::string capability_str = prop.gcnArchName;
#elif defined(NOA_ENABLE_SYCL)
            const char* backend_name = "sycl";
            std::string name = prop.get_info<sycl::info::device::name>();
            std::string capability_str = "n/a";
#endif

            return fmt::format(
                "{}:{}:\n"
                "    Name: {}\n"
                "    Memory: {}MB / {}MB\n"
                "    Capabilities: {}\n"
                "    Runtime version: {}\n"
                "    Driver version: {}",
                backend_name, id(), name,
                (mem.total - mem.free) / 1048576, mem.total / 1048576,
                capability_str, runtime_version(), driver_version()
            );
        }

        /// Gets resource limits for the current device.
        [[nodiscard]] auto limit(DeviceLimit resource_limit) const -> usize {
#if defined(NOA_ENABLE_CUDA)
            usize limit;
            const Device previous_current = current();
            set_current(*this);
            check(cudaDeviceGetLimit(&limit, resource_limit));
            set_current(previous_current);
            return limit;
#elif defined(NOA_ENABLE_HIP)
            // Only a subset of limits is supported on HIP, unsupported ones return hipErrorUnsupportedLimit.
            usize limit;
            const Device previous_current = current();
            set_current(*this);
            check(hipDeviceGetLimit(&limit, resource_limit));
            set_current(previous_current);
            return limit;
#elif defined(NOA_ENABLE_SYCL)
            (void) resource_limit;
            panic("Not supported with SYCL"); // TODO
#endif
        }

        [[nodiscard]] auto get() const noexcept -> i32 { return m_id; }
        [[nodiscard]] auto id() const noexcept -> i32 { return m_id; }
#if defined(NOA_ENABLE_SYCL)
        [[nodiscard]] auto get_sycl_device() const -> sycl::device {
            return details::sycl_devices()[static_cast<usize>(m_id)];
        }
#endif

    public: // Static functions
        /// Returns the number of compute-capable devices.
        static auto count() -> i32 {
#if defined(NOA_ENABLE_CUDA)
            i32 count{};
            check(cudaGetDeviceCount(&count));
            return count;
#elif defined(NOA_ENABLE_HIP)
            i32 count{};
            check(hipGetDeviceCount(&count));
            return count;
#elif defined(NOA_ENABLE_SYCL)
            return static_cast<i32>(details::sycl_devices().size());
#endif
        }

        /// Whether there's any CUDA capable device.
        static auto is_any() -> bool {
            return count() != 0;
        }

        /// Returns the number of compute-capable devices.
        static auto all() -> std::vector<Device> {
            std::vector<Device> devices{};
            const auto count = static_cast<usize>(Device::count());
            devices.reserve(count);
            for (usize id = 0; id < count; ++id)
                devices.emplace_back(id);
            return devices;
        }

        /// Returns the device on which the active host thread executes the device code.
        /// The default device is the first device, i.e. device with ID=0.
        static auto current() -> Device {
            auto device = Device(0, Unchecked{});
#if defined(NOA_ENABLE_CUDA)
            check(cudaGetDevice(&device.m_id));
#elif defined(NOA_ENABLE_HIP)
            check(hipGetDevice(&device.m_id));
#elif defined(NOA_ENABLE_SYCL)
            device.m_id = details::sycl_current_device_id();
#endif
            return device;
        }

        /// Sets device as the current device for the calling host thread.
        /// \note (CUDA)
        ///     "Any device memory subsequently allocated from this host thread [...] will be physically resident
        ///     on device. Any host memory allocated from this host thread [...] will have its lifetime
        ///     associated with device. Any streams or events created from this host thread will be associated
        ///     with device. Any kernels launched from this host thread [...] will be executed on device.
        ///     This call may be made from any host thread, to any device, and at any time. This function will do
        ///     no synchronization with the previous or new device, and should be considered a very low overhead call".
        static void set_current(Device device) {
#if defined(NOA_ENABLE_CUDA)
            check(cudaSetDevice(device.m_id));
#elif defined(NOA_ENABLE_HIP)
            check(hipSetDevice(device.m_id));
#elif defined(NOA_ENABLE_SYCL)
            details::sycl_set_current_device_id(device.m_id);
#endif
        }

        /// Gets the device with the most free memory available for allocation.
        static auto most_free() -> Device {
            auto most_free = Device(0, Unchecked{});
            usize available_mem{};
            for (auto& device: all()) {
                const usize dev_available_mem = device.memory().free;
                if (dev_available_mem > available_mem) {
                    most_free = device;
                    available_mem = dev_available_mem;
                }
            }
            return most_free;
        }

    private:
        static void validate_(i32 id) {
            const isize count = Device::count();
            check(id < count, "Invalid device ID. Got ID={}, count:{}", id, count);
        }

    private:
        i32 m_id{};
    };

    inline bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }
}

namespace noa::gpu {
    // A device that sets itself as the current device for the remainder of the scope.
    class DeviceGuard : public Device {
    public:
        template<typename... Args>
        explicit DeviceGuard(Args... args) : Device(args...), m_previous_current(current()) {
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
}
