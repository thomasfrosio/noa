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
#include "noa/common/string/Convert.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Version.h"

namespace noa::cuda {
    struct device_memory_info_t { size_t free; size_t total; };
    struct device_capability_t { int major; int minor; };

    /// A CUDA device.
    class Device {
    public:
        Device() = default;

        /// Creates a CUDA device using the device ID.
        NOA_HOST explicit Device(int device_id) : m_id(device_id) {};

        /// Creates a CUDA device using the device name.
        /// The device name should be of the form, "cuda:N", where N is the device ID.
        NOA_HOST explicit Device(std::string_view name) : m_id(parseID_(name)) {};

        NOA_HOST [[nodiscard]] int get() const noexcept { return m_id; }
        NOA_HOST [[nodiscard]] int id() const noexcept { return m_id; }

        /// Suspends execution until all previously-scheduled tasks on the specified device have concluded.
        /// \see See corresponding static function for more details.
        NOA_HOST void synchronize() const {
            Device::synchronize(*this);
        }

        /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
        /// \see See corresponding static function for more details.
        NOA_HOST void reset() const {
            Device::reset(*this);
        }

        /// Retrieves the properties of the device.
        [[nodiscard]] NOA_HOST cudaDeviceProp properties() const {
            NOA_PROFILE_FUNCTION();
            cudaDeviceProp properties{};
            NOA_THROW_IF(cudaGetDeviceProperties(&properties, m_id));
            return properties;
        }

        /// Retrieves an attribute (one of the many that composes a device_prop_t) from a device.
        /// \warning The following attributes require PCIe reads and are therefore much slower to get:
        ///          cudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate,
        ///          and cudaDevAttrSingleToDoublePrecisionPerfRatio
        [[nodiscard]] NOA_HOST int attribute(cudaDeviceAttr attribute) const {
            int attribute_value;
            NOA_THROW_IF(cudaDeviceGetAttribute(&attribute_value, attribute, m_id));
            return attribute_value;
        }

        /// Gets the device's hardware architecture generation numeric designator.
        [[nodiscard]] NOA_HOST int architecture() const {
            int major;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            return major;
        }

        /// Gets the device's compute capability (major, minor) numeric designator.
        [[nodiscard]] NOA_HOST device_capability_t capability() const {
            int major, minor;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_id));
            NOA_THROW_IF(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, m_id));
            return {major, minor};
        }

        /// Returns the free and total amount of memory available for allocation by the device, in bytes.
        [[nodiscard]] NOA_HOST device_memory_info_t memory() const {
            NOA_PROFILE_FUNCTION();
            size_t mem_free, mem_total;

            Device previous_current = Device::getCurrent();
            Device::setCurrent(*this);
            NOA_THROW_IF(cudaMemGetInfo(&mem_free, &mem_total));
            Device::setCurrent(previous_current);
            return {mem_free, mem_total};
        }

        /// Retrieves a summary of the device. This is quite an "expensive" operation.
        [[nodiscard]] NOA_HOST std::string summary() const {
            NOA_PROFILE_FUNCTION();
            cudaDeviceProp prop = properties();
            device_memory_info_t mem = memory();
            auto version_formatter = [](int version) -> std::pair<int, int> {
                int major = version / 1000;
                int minor = version / 10 - major * 100;
                return {major, minor};
            };
            auto[runtime_major, runtime_minor] = version_formatter(versionRuntime());
            auto[driver_major, driver_minor] = version_formatter(versionDriver());

            return string::format("cuda:{}:\n"
                                  "    Name: {}\n"
                                  "    Memory: {} MB / {} MB\n"
                                  "    Capabilities: {}.{}\n"
                                  "    Runtime version: {}.{}\n"
                                  "    Driver version: {}.{}",
                                  id(), prop.name,
                                  (mem.total - mem.free) / 100000, mem.total / 100000,
                                  prop.major, prop.minor,
                                  runtime_major, runtime_minor,
                                  driver_major, driver_minor);
        }

        /// Gets resource limits for the current device.
        [[nodiscard]] NOA_HOST size_t limit(cudaLimit resource_limit) const {
            size_t limit;
            Device previous_current = Device::getCurrent();
            Device::setCurrent(*this);
            NOA_THROW_IF(cudaDeviceGetLimit(&limit, resource_limit));
            Device::setCurrent(previous_current);
            return limit;
        }

    public: // Static functions
        /// Returns the number of compute-capable devices.
        NOA_HOST static size_t getCount() {
            int count{};
            NOA_THROW_IF(cudaGetDeviceCount(&count));
            return static_cast<size_t>(count);
        }

        /// Returns the number of compute-capable devices.
        NOA_HOST static std::vector<Device> getAll() {
            std::vector<Device> devices;
            size_t count = getCount();
            devices.reserve(count);
            for (int id = 0; id < static_cast<int>(count); ++id)
                devices.emplace_back(id);
            return devices;
        }

        /// Returns the device on which the active host thread executes the device code.
        NOA_HOST static Device getCurrent() {
            Device device;
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
        NOA_HOST static void setCurrent(Device device) {
            NOA_THROW_IF(cudaSetDevice(device.m_id));
        }

        /// Gets the device with the most free memory available for allocation.
        NOA_HOST static Device getMostFree() {
            NOA_PROFILE_FUNCTION();
            Device most_free;
            size_t available_mem{0};
            for (auto& device: getAll()) {
                size_t dev_available_mem = device.memory().free;
                if (dev_available_mem > available_mem) {
                    most_free = device;
                    available_mem = dev_available_mem;
                }
            }
            return most_free;
        }

        /// Suspends execution until all previously-scheduled tasks on the specified device (all contexts and streams)
        /// have concluded. Depending on the host synchronization scheduling policy set for this device, the calling
        /// thread will either yield, spin or block until this completion.
        /// \note By default behavior for host synchronization is based on the number of active CUDA contexts in the
        ///       process C and the number of logical processors in the system P. If C > P, then CUDA will yield to
        ///       other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results
        ///       and actively spin on the processor. Yielding can increase latency when waiting for the device, but can
        ///       increase the performance of CPU threads performing work in parallel with the device. Spinning is the
        ///       other way around.
        NOA_HOST static void synchronize(Device device) {
            NOA_PROFILE_FUNCTION();
            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaDeviceSynchronize());
            Device::setCurrent(previous_current);
        }

        /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
        /// Any subsequent API call to this device will reinitialize the device.
        /// \warning This function will reset the device immediately. It is the caller's responsibility to ensure that
        ///          the device is not being accessed by any other host threads from the process when this function
        ///          is called.
        NOA_HOST static void reset(Device device) {
            NOA_PROFILE_FUNCTION();
            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaDeviceReset());
            Device::setCurrent(previous_current);
        }

    private:
        int m_id{};

        NOA_HOST static int parseID_(std::string_view name) {
            try {
                if (name.size() < 6 || name.rfind("cuda:", 0))
                    throw;
                return string::toInt<int>(name.data() + 5);
            } catch (...) {
                NOA_THROW("Could not convert {} into a cuda device name. "
                          "Should be \"cuda:N\", where N is an integer starting from 0",
                          name);
            }
        }
    };

    NOA_IH bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }

    /// Retrieves the device's human readable name.
    NOA_IH std::ostream& operator<<(std::ostream& os, Device device) {
        os << string::format("cuda:{}", device.id());
        return os;
    }

    /// Sets the device as the current device for the remainder of the scope in which this object is invoked,
    /// and changes it back to the previous device when exiting the scope.
    class DeviceCurrentScope : public Device {
    private:
        Device m_previous_current;
    public:
        NOA_HOST explicit DeviceCurrentScope(Device device)
                : Device(device), m_previous_current(Device::getCurrent()) {
            Device::setCurrent(device);
        }

        NOA_HOST ~DeviceCurrentScope() {
            Device::setCurrent(m_previous_current);
        }
    };

    NOA_IH bool operator==(const Device& lhs, const DeviceCurrentScope& rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator==(const DeviceCurrentScope& lhs, const Device& rhs) { return lhs.id() == rhs.id(); }

    NOA_IH bool operator!=(const Device& lhs, const DeviceCurrentScope& rhs) { return lhs.id() != rhs.id(); }
    NOA_IH bool operator!=(const DeviceCurrentScope& lhs, const Device& rhs) { return lhs.id() != rhs.id(); }
}
