/// \file noa/gpu/cuda/Device.h
/// \brief CUDA devices.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/string/Format.h"
#include "noa/common/string/Parse.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Version.h"

namespace noa::cuda {
    struct DeviceMemory { size_t free; size_t total; }; // bytes
    struct DeviceCapability { int major; int minor; };

    /// A CUDA device.
    class Device {
    public:
        /// Creates the current device.
        Device() : m_id(Device::current().m_id) {}

        /// Creates a CUDA device from an ID.
        constexpr explicit Device(int id, bool unsafe = false) : m_id(id) {
            if (!unsafe)
                validate_(m_id);
        };

        /// Creates a CUDA device from a name.
        /// The device name should be of the form, "cuda:N", where N is a device ID.
        explicit Device(std::string_view name, bool unsafe = false) : m_id(parse_(name)) {
            if (!unsafe)
                validate_(m_id);
        };

    public:
        /// Suspends execution until all previously-scheduled tasks on the specified device (all contexts and streams)
        /// have concluded. Depending on the host synchronization scheduling policy set for this device, the calling
        /// thread will either yield, spin or block until this completion.
        /// \note By default behavior for host synchronization is based on the number of active CUDA contexts in the
        ///       process C and the number of logical processors in the system P. If C > P, then CUDA will yield to
        ///       other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results
        ///       and actively spin on the processor. Yielding can increase latency when waiting for the device, but can
        ///       increase the performance of CPU threads performing work in parallel with the device. Spinning is the
        ///       other way around.
        void synchronize() const {
            NOA_PROFILE_FUNCTION();
            Device previous_current = Device::current();
            Device::current(*this);
            NOA_THROW_IF(cudaDeviceSynchronize());
            Device::current(previous_current);
        }

        /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
        /// Any subsequent API call to this device will reinitialize the device.
        /// \warning This function will reset the device immediately. It is the caller's responsibility to ensure that
        ///          the device is not being accessed by any other host threads from the process when this function
        ///          is called.
        void reset() const;

        /// Retrieves the properties of the device.
        [[nodiscard]] cudaDeviceProp properties() const {
            NOA_PROFILE_FUNCTION();
            cudaDeviceProp properties{};
            NOA_THROW_IF(cudaGetDeviceProperties(&properties, m_id));
            return properties;
        }

        /// Retrieves an attribute (one of the many that composes a device_prop_t) from a device.
        /// \warning The following attributes require PCIe reads and are therefore much slower to get:
        ///          cudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate,
        ///          and cudaDevAttrSingleToDoublePrecisionPerfRatio
        [[nodiscard]] int attribute(cudaDeviceAttr attribute) const {
            int attribute_value;
            NOA_THROW_IF(cudaDeviceGetAttribute(&attribute_value, attribute, m_id));
            return attribute_value;
        }

        /// Gets the device's hardware architecture generation numeric designator.
        [[nodiscard]] int architecture() const {
            int major;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            return major;
        }

        /// Gets the device's compute capability (major, minor) numeric designator.
        [[nodiscard]] DeviceCapability capability() const {
            int major, minor;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            NOA_THROW_IF(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, m_id));
            return {major, minor};
        }

        /// Returns the free and total amount of memory available for allocation by the device, in bytes.
        [[nodiscard]] DeviceMemory memory() const {
            NOA_PROFILE_FUNCTION();
            size_t mem_free, mem_total;

            Device previous_current = Device::current();
            Device::current(*this);
            NOA_THROW_IF(cudaMemGetInfo(&mem_free, &mem_total));
            Device::current(previous_current);
            return {mem_free, mem_total};
        }

        /// Retrieves a summary of the device. This is quite an "expensive" operation.
        [[nodiscard]] std::string summary() const {
            NOA_PROFILE_FUNCTION();
            cudaDeviceProp prop = properties();
            DeviceMemory mem = memory();
            auto version_formatter = [](int version) -> std::pair<int, int> {
                int major = version / 1000;
                int minor = version / 10 - major * 100;
                return {major, minor};
            };
            auto[runtime_major, runtime_minor] = version_formatter(versionRuntime());
            auto[driver_major, driver_minor] = version_formatter(versionDriver());

            return string::format("cuda:{}:\n"
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

        /// Gets resource limits for the current device.
        [[nodiscard]] size_t limit(cudaLimit resource_limit) const {
            size_t limit;
            Device previous_current = Device::current();
            Device::current(*this);
            NOA_THROW_IF(cudaDeviceGetLimit(&limit, resource_limit));
            Device::current(previous_current);
            return limit;
        }

        [[nodiscard]] int get() const noexcept { return m_id; }
        [[nodiscard]] int id() const noexcept { return m_id; }

    public: // Static functions
        /// Returns the number of compute-capable devices.
        static size_t count() {
            int count{};
            NOA_THROW_IF(cudaGetDeviceCount(&count));
            return static_cast<size_t>(count);
        }

        /// Whether there's any CUDA capable device.
        static bool any() {
            return count() != 0;
        }

        /// Returns the number of compute-capable devices.
        static std::vector<Device> all() {
            std::vector<Device> devices;
            size_t count = Device::count();
            devices.reserve(count);
            for (int id = 0; id < static_cast<int>(count); ++id)
                devices.emplace_back(id);
            return devices;
        }

        /// Returns the device on which the active host thread executes the device code.
        /// The default device is the first device, i.e. device with ID=0.
        static Device current() {
            Device device(0, true);
            NOA_THROW_IF(cudaGetDevice(&device.m_id));
            return device;
        }

        /// Sets device as the current device for the calling host thread.
        /// \details "Any device memory subsequently allocated from this host thread [...] will be physically resident
        ///          on \p device. Any host memory allocated from this host thread [...] will have its lifetime
        ///          associated with \p device. Any streams or events created from this host thread will be associated
        ///          with \p device. Any kernels launched from this host thread [...] will be executed on \p device.
        ///          This call may be made from any host thread, to any device, and at any time. This function will do
        ///          no synchronization with the previous or new device, and should be considered a very low overhead
        ///          call".
        static void current(Device device) {
            NOA_THROW_IF(cudaSetDevice(device.m_id));
        }

        /// Gets the device with the most free memory available for allocation.
        static Device mostFree() {
            NOA_PROFILE_FUNCTION();
            Device most_free(0, true);
            size_t available_mem{0};
            for (auto& device: all()) {
                size_t dev_available_mem = device.memory().free;
                if (dev_available_mem > available_mem) {
                    most_free = device;
                    available_mem = dev_available_mem;
                }
            }
            return most_free;
        }

    private:
        int m_id{};

        static int parse_(std::string_view name) {
            name = string::trim(string::lower(name));
            const size_t length = name.length();

            if (string::startsWith(name, "cuda")) {
                if (length == 4)
                    return 0;
                else if (length >= 6 && name[4] == ':')
                    return string::toInt<int>(std::string{name.data() + 5});
            }

            NOA_THROW("Failed to parse CUDA device name:\"{}\"", name);
        }

        static void validate_(int id) {
            const size_t count = Device::count();
            if (static_cast<size_t>(id) + 1 > count)
                NOA_THROW("Invalid device ID. Got ID:{}, count:{}", id, count);
        }
    };

    NOA_IH bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }
}

namespace noa::cuda {
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
