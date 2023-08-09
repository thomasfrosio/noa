#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include "noa/core/Definitions.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/string/Parse.hpp"

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/utils/Version.hpp"

namespace noa::cuda {
    struct DeviceMemory { size_t free; size_t total; }; // bytes
    struct DeviceCapability { int major; int minor; };

    // A CUDA device.
    class Device {
    public:
        struct DeviceUnchecked{};

    public:
        // Creates the current device.
        Device() : m_id(Device::current().m_id) {}

        // Creates a CUDA device from an ID.
        template<typename Int = i64, typename = std::enable_if_t<nt::is_restricted_int_v<Int>>>
        constexpr explicit Device(Int id) : m_id(static_cast<i32>(id)) {
            validate_(m_id);
        }

        // Creates a CUDA device from a name.
        // The device name should be of the form, "cuda:N", where N is a device ID.
        explicit Device(std::string_view name) : m_id(parse_id_(name)) {
            validate_(m_id);
        }

        template<typename Int = i64, typename = std::enable_if_t<nt::is_restricted_int_v<Int>>>
        constexpr explicit Device(Int id, DeviceUnchecked) : m_id(static_cast<i32>(id)) {}

        explicit Device(std::string_view name, DeviceUnchecked) : m_id(parse_id_(name)) {}

    public:
        // Suspends execution until all previously-scheduled tasks on the specified device (all contexts and streams)
        // have concluded. Depending on the host synchronization scheduling policy set for this device, the calling
        // thread will either yield, spin or block until this completion.
        // By default, behavior for host synchronization is based on the number of active CUDA contexts in the
        // process C and the number of logical processors in the system P. If C > P, then CUDA will yield to
        // other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results
        // and actively spin on the processor. Yielding can increase latency when waiting for the device, but can
        // increase the performance of CPU threads performing work in parallel with the device. Spinning is the
        // other way around.
        void synchronize() const {
            const Device previous_current = Device::current();
            Device::set_current(*this);
            NOA_THROW_IF(cudaDeviceSynchronize());
            Device::set_current(previous_current);
        }

        // Explicitly synchronizes, destroys and cleans up all resources associated with the current device in the
        // current process. Any subsequent API call to this device will reinitialize the device.
        // It is the caller's responsibility to ensure that resources (streams, pinned|device arrays,
        // CUDA arrays, textures) attached to that device are destructed before calling this function.
        // The library's internal data will be handled automatically, e.g. FFT plans.
        void reset() const;

        // Retrieves the properties of the device.
        [[nodiscard]] cudaDeviceProp properties() const {
            cudaDeviceProp properties{};
            NOA_THROW_IF(cudaGetDeviceProperties(&properties, m_id));
            return properties;
        }

        // Retrieves an attribute (one of the many that composes a device_prop_t) from a device.
        // The following attributes require PCIe reads and are therefore much slower to get:
        // cudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate,
        // and cudaDevAttrSingleToDoublePrecisionPerfRatio
        [[nodiscard]] int attribute(cudaDeviceAttr attribute) const {
            int attribute_value;
            NOA_THROW_IF(cudaDeviceGetAttribute(&attribute_value, attribute, m_id));
            return attribute_value;
        }

        // Gets the device's hardware architecture generation numeric designator.
        [[nodiscard]] int architecture() const {
            int major;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            return major;
        }

        // Gets the device's compute capability (major, minor) numeric designator.
        [[nodiscard]] DeviceCapability capability() const {
            int major, minor;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            NOA_THROW_IF(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, m_id));
            return {major, minor};
        }

        // Returns the free and total amount of memory available for allocation by the device, in bytes.
        [[nodiscard]] DeviceMemory memory() const {
            size_t mem_free, mem_total;

            const Device previous_current = Device::current();
            Device::set_current(*this);
            NOA_THROW_IF(cudaMemGetInfo(&mem_free, &mem_total));
            Device::set_current(previous_current);
            return {mem_free, mem_total};
        }

        // Retrieves a summary of the device. This is quite an "expensive" operation.
        [[nodiscard]] std::string summary() const {
            const cudaDeviceProp prop = properties();
            const DeviceMemory mem = memory();
            const auto version_formatter = [](int version) -> std::pair<int, int> {
                const int major = version / 1000;
                const int minor = version / 10 - major * 100;
                return {major, minor};
            };
            const auto[runtime_major, runtime_minor] = version_formatter(version_runtime());
            const auto[driver_major, driver_minor] = version_formatter(version_driver());

            return noa::string::format(
                    "cuda:{}:\n"
                    "    Name: {}\n"
                    "    Memory: {}MB / {}MB\n"
                    "    Capabilities: {}.{}\n"
                    "    Runtime version: {}.{}\n"
                    "    Driver version: {}.{}",
                    id(), prop.name,
                    (mem.total - mem.free) / 1048576, mem.total / 1048576,
                    prop.major, prop.minor,
                    runtime_major, runtime_minor,
                    driver_major, driver_minor);
        }

        // Gets resource limits for the current device.
        [[nodiscard]] size_t limit(cudaLimit resource_limit) const {
            size_t limit;
            const Device previous_current = Device::current();
            Device::set_current(*this);
            NOA_THROW_IF(cudaDeviceGetLimit(&limit, resource_limit));
            Device::set_current(previous_current);
            return limit;
        }

        [[nodiscard]] i32 get() const noexcept { return m_id; }
        [[nodiscard]] i32 id() const noexcept { return m_id; }

    public: // Static functions
        // Returns the number of compute-capable devices.
        static i32 count() {
            int count{};
            NOA_THROW_IF(cudaGetDeviceCount(&count));
            return count;
        }

        // Whether there's any CUDA capable device.
        static bool is_any() {
            return count() != 0;
        }

        // Returns the number of compute-capable devices.
        static std::vector<Device> all() {
            std::vector<Device> devices{};
            const auto count = static_cast<size_t>(Device::count());
            devices.reserve(count);
            for (size_t id = 0; id < count; ++id)
                devices.emplace_back(id);
            return devices;
        }

        // Returns the device on which the active host thread executes the device code.
        // The default device is the first device, i.e. device with ID=0.
        static Device current() {
            Device device(0, DeviceUnchecked{});
            NOA_THROW_IF(cudaGetDevice(&device.m_id));
            return device;
        }

        // Sets device as the current device for the calling host thread.
        // "Any device memory subsequently allocated from this host thread [...] will be physically resident
        // on device. Any host memory allocated from this host thread [...] will have its lifetime
        // associated with device. Any streams or events created from this host thread will be associated
        // with device. Any kernels launched from this host thread [...] will be executed on device.
        // This call may be made from any host thread, to any device, and at any time. This function will do
        // no synchronization with the previous or new device, and should be considered a very low overhead
        // call".
        static void set_current(Device device) {
            NOA_THROW_IF(cudaSetDevice(device.m_id));
        }

        // Gets the device with the most free memory available for allocation.
        static Device most_free() {
            Device most_free(0, DeviceUnchecked{});
            size_t available_mem{0};
            for (auto& device: all()) {
                const size_t dev_available_mem = device.memory().free;
                if (dev_available_mem > available_mem) {
                    most_free = device;
                    available_mem = dev_available_mem;
                }
            }
            return most_free;
        }

    private:
        static i32 parse_id_(std::string_view name) {
            std::string str_ = noa::string::lower(noa::string::trim(name));

            if (!noa::string::starts_with(str_, "cuda"))
                NOA_THROW("Failed to parse CUDA device \"{}\"", str_);

            i32 id{};
            const size_t length = str_.length();
            if (length == 4) {
                id = 0;
            } else if (length >= 6 && str_[4] == ':') {
                i32 error{};
                id = noa::string::parse<i32>(std::string{str_.data() + 5}, error);
                if (error)
                    NOA_THROW("Failed to parse the CUDA device ID. {}",
                              noa::string::parse_error_message<i32>(str_, error));
            } else {
                NOA_THROW("Failed to parse CUDA device \"{}\"", str_);
            }
            return id;
        }

        static void validate_(i32 id) {
            const i64 count = Device::count();
            if (id + 1 > count)
                NOA_THROW("Invalid device ID. Got ID:{}, count:{}", id, count);
        }

    private:
        i32 m_id{};
    };

    inline bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    inline bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }
}

namespace noa::cuda {
    // A device that sets itself as the current device for the remainder of the scope.
    class DeviceGuard : public Device {
    public:
        template<typename ... Args>
        explicit DeviceGuard(Args&& ...args)
                : Device(std::forward<Args>(args)...),
                  m_previous_current(Device::current()) {
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
}
