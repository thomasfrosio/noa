#pragma once

#include "noa/gpu/cuda/Base.h"

namespace Noa::CUDA {
    /** A device and a namespace-like. */
    class Device {
    public: // Type definitions
        using id_t = int;
        using prop_t = cudaDeviceProp;
        using attr_t = cudaDeviceAttr;
        using limit_t = cudaLimit;
        struct meminfo_t { size_t free; size_t total; };
        struct capability_t { int major; int minor; };

    private: // Member variables
        Device::id_t m_id{};

    public: // Member functions
        NOA_IH Device() = default;
        NOA_IH explicit Device(Device::id_t device_id) : m_id(device_id) {};
        NOA_IH Device::id_t get() const noexcept { return m_id; }
        NOA_IH Device::id_t id() const noexcept { return m_id; }

    public: // Static functions
        /** Returns the number of compute-capable devices. */
        NOA_IH static size_t getCount() {
            int count{};
            NOA_THROW_IF(cudaGetDeviceCount(&count));
            return static_cast<size_t>(count);
        }

        /** Returns the number of compute-capable devices. */
        NOA_IH static std::vector<Device> getAll() {
            std::vector<Device> devices;
            size_t count = getCount();
            devices.reserve(count);
            for (size_t id = 0; id < count; ++id) {
                devices.emplace_back(static_cast<id_t>(id));
            }
            return devices;
        }

        /** Returns the device on which the active host thread executes the device code. */
        NOA_IH static Device getCurrent() {
            Device device;
            NOA_THROW_IF(cudaGetDevice(&device.m_id));
            return device;
        }

        /**
         * Sets device as the current device for the calling host thread.
         * @details Any device memory subsequently allocated from this host thread [...] will be physically resident
         *          on @a device. Any host memory allocated from this host thread using [...] will have its lifetime
         *          associated with @a device. Any streams or events created from this host thread will be associated
         *          with @a device. Any kernels launched from this host thread [...] will be executed on @a device.
         *          This call may be made from any host thread, to any device, and at any time. This function will do
         *          no synchronization with the previous or new device, and should be considered a very low overhead
         *          call.
         */
        NOA_IH static void setCurrent(Device device) {
            NOA_THROW_IF(cudaSetDevice(device.m_id));
        }

        /** Retrieves the properties of @a device. */
        NOA_IH static Device::prop_t getProperties(Device device) {
            Device::prop_t properties;
            NOA_THROW_IF(cudaGetDeviceProperties(&properties, device.m_id));
            return properties;
        }

        /**
         * Retrieves an attribute (one of the many that composes a device_prop_t) from a device.
         * @warning The following attributes require PCIe reads and are therefore much slower to get:
         *          cudaDevAttrClockRate, cudaDevAttrKernelExecTimeout, cudaDevAttrMemoryClockRate,
         *          and cudaDevAttrSingleToDoublePrecisionPerfRatio
         */
        NOA_IH static int getAttribute(Device::attr_t attribute, Device device) {
            int attribute_value;
            NOA_THROW_IF(cudaDeviceGetAttribute(&attribute_value, attribute, device.m_id));
            return attribute_value;
        }

        /** Gets the device's hardware architecture generation numeric designator. */
        NOA_IH static int getArchitecture(Device device) {
            int major;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.m_id));
            return major;
        }

        /** Gets the device's compute capability (major, minor) numeric designator. */
        NOA_IH static Device::capability_t getComputeCapability(Device device) {
            int major, minor;
            NOA_THROW_IF(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device.m_id));
            NOA_THROW_IF(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device.m_id));
            return {major, minor};
        }

        /** Gets the latest version of CUDA supported by the driver. Format 1000 * major + 10 * minor. */
        NOA_IH static int getVersionDriver() {
            int version;
            NOA_THROW_IF(cudaDriverGetVersion(&version));
            return version;
        }

        /** Gets the CUDA runtime version. Format 1000 * major + 10 * minor. */
        NOA_IH static int getVersionRuntime() {
            int version;
            NOA_THROW_IF(cudaRuntimeGetVersion(&version));
            return version;
        }

        /** Returns the free and total amount of memory available for allocation by the device, in bytes. */
        NOA_IH static Device::meminfo_t getMemoryInfo(Device device) {
            size_t mem_free, mem_total;

            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaMemGetInfo(&mem_free, &mem_total));
            Device::setCurrent(previous_current);
            return {mem_free, mem_total};
        }

        /** Gets the device with the most free memory available for allocation. */
        NOA_IH static Device getMostFree() {
            Device most_free;
            size_t available_mem{0};
            for (auto& device: getAll()) {
                size_t dev_available_mem = getMemoryInfo(device).free;
                if (dev_available_mem > available_mem) {
                    most_free = device;
                    available_mem += dev_available_mem;
                }
            }
            return most_free;
        }

        /** Retrieves a summary of the device. This is quite an expensive operation. */
        NOA_IH static std::string getSummary(Device device) {
            Device::prop_t properties = Device::getProperties(device);
            Device::meminfo_t memory = Device::getMemoryInfo(device);
            auto version_formatter = [](int version) -> std::pair<int, int> {
                int major = version / 1000;
                int minor = version / 10 - major * 100;
                return {major, minor};
            };
            auto [runtime_major, runtime_minor] = version_formatter(getVersionRuntime());
            auto [driver_major, driver_minor] = version_formatter(getVersionDriver());

            return String::format("Device {}:\n"
                                  "    Name: {}\n"
                                  "    Memory: {} MB / {} MB\n"
                                  "    Capabilities: {}.{}\n"
                                  "    Runtime version: {}.{}\n"
                                  "    Driver version: {}.{}",
                                  device.id(), properties.name,
                                  (memory.total - memory.free) / 100000, memory.total / 100000,
                                  properties.major, properties.minor,
                                  runtime_major, runtime_minor,
                                  driver_major, driver_minor);
        }

        /** Gets resource limits for the current device. */
        NOA_IH static size_t getLimit(Device::limit_t resource_limit, Device device) {
            size_t limit;
            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaDeviceGetLimit(&limit, resource_limit));
            Device::setCurrent(previous_current);
            return limit;
        }

        /**
         * Suspends execution until all previously-scheduled tasks on the specified device (all contexts and streams)
         * have concluded. Depending on the host synchronization scheduling policy set for this device, the calling
         * thread will either yield, spin or block until this completion.
         * @note By default behavior for host synchronization is based on the number of active CUDA contexts in the
         *       process C and the number of logical processors in the system P. If C > P, then CUDA will yield to
         *       other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results
         *       and actively spin on the processor. Yielding can increase latency when waiting for the device, but can
         *       increase the performance of CPU threads performing work in parallel with the device. Spinning is the
         *       other way around.
         */
        NOA_IH static void synchronize(Device device) {
            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaDeviceSynchronize());
            Device::setCurrent(previous_current);
        }

        /**
         * Explicitly destroys and cleans up all resources associated with the current device in the current process.
         * Any subsequent API call to this device will reinitialize the device.
         * @warning This function will reset the device immediately. It is the caller's responsibility to ensure that
         *          the device is not being accessed by any other host threads from the process when this function
         *          is called.
         */
        NOA_IH static void reset(Device device) {
            Device previous_current = Device::getCurrent();
            Device::setCurrent(device);
            NOA_THROW_IF(cudaDeviceReset());
            Device::setCurrent(previous_current);
        }
    };

    NOA_IH bool operator==(Device lhs, Device rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator!=(Device lhs, Device rhs) { return lhs.id() != rhs.id(); }

    /** Retrieves the device's human readable name. */
    NOA_IH static std::string toString(Device device) { return Device::getProperties(device).name; }

    /**
     * Sets the device as the current device for the remainder of the scope in which this object is invoked,
     * and changes it back to the previous device when exiting the scope.
     */
    class DeviceCurrentScope : public Device {
    private:
        Device m_previous_current;
    public:
        NOA_IH explicit DeviceCurrentScope(Device device)
                : Device(device), m_previous_current(Device::getCurrent()) {
            Device::setCurrent(device);
        }

        NOA_IH ~DeviceCurrentScope() {
            Device::setCurrent(m_previous_current);
        }
    };

    // Don't slice... for clang-tidy.
    NOA_IH bool operator==(Device lhs, DeviceCurrentScope rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator==(DeviceCurrentScope lhs, Device rhs) { return lhs.id() != rhs.id(); }

    NOA_IH bool operator!=(Device lhs, DeviceCurrentScope rhs) { return lhs.id() == rhs.id(); }
    NOA_IH bool operator!=(DeviceCurrentScope lhs, Device rhs) { return lhs.id() != rhs.id(); }
}

template<>
struct fmt::formatter<Noa::CUDA::Device> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(Noa::CUDA::Device device, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::CUDA::toString(device), ctx);
    }
};

std::ostream& operator<<(std::ostream& os, Noa::CUDA::Device device) {
    os << Noa::CUDA::toString(device);
    return os;
}
