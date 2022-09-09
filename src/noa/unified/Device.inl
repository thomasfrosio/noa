#ifndef NOA_UNIFIED_DEVICE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

namespace noa {
    inline constexpr Device::Device(Device::Type type, int id, bool unsafe)
            : m_id(type == Type::CPU ? -1 : id) {
        if (!unsafe)
            validate_(type, m_id);
    }

    inline Device::Device(std::string_view name, bool unsafe) {
        std::string str_ = string::lower(string::trim(name));
        const size_t length = name.length();

        Type type;
        int error{0};
        if (string::startsWith(str_, "cpu")) {
            if (length == 3) {
                m_id = -1;
                type = Type::CPU;
            } else {
                NOA_THROW("CPU device name \"{}\" is not supported", str_);
            }

        } else if (string::startsWith(str_, "gpu")) {
            if (length == 3) {
                m_id = 0;
            } else if (length >= 5 && str_[3] == ':') {
                m_id = string::parse<int>(std::string(str_.data() + 4), error);
            } else {
                NOA_THROW("GPU device name \"{}\" is not supported", str_);
            }
            type = Type::GPU;

        } else if (string::startsWith(str_, "cuda")) {
            if (length == 4) {
                m_id = 0;
            } else if (length >= 6 && str_[4] == ':') {
                m_id = string::parse<int>(std::string(str_.data() + 5), error);
            } else {
                NOA_THROW("CUDA device name \"{}\" is not supported", str_);
            }
            type = Type::GPU;
        } else {
            NOA_THROW("Failed to parse \"{}\" as a valid device name", str_);
        }

        if (error)
            NOA_THROW("Failed to parse the device ID. {}", string::parseErrorMessage<int>(str_, error));

        if (!unsafe)
            validate_(type, m_id);
    }

    inline std::string Device::summary() const {
        if (cpu()) {
            return cpu::Device::summary();
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::Device(this->id(), true).summary();
            #else
            return {};
            #endif
        }
    }

    inline DeviceMemory Device::memory() const {
        if (cpu()) {
            const cpu::DeviceMemory mem_info = cpu::Device::memory();
            return {mem_info.total, mem_info.free};
        } else {
            #ifdef NOA_ENABLE_CUDA
            const cuda::DeviceMemory mem_info = cuda::Device(this->id(), true).memory();
            return {mem_info.total, mem_info.free};
            #else
            return {};
            #endif
        }
    }

    inline void Device::memoryThreshold(size_t threshold_bytes) const {
        if (gpu()) {
            #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
            cuda::memory::Pool(cuda::Device(id(), true)).threshold(threshold_bytes);
            #else
            (void) threshold_bytes;
            return;
            #endif
        }
    }

    inline void Device::memoryTrim(size_t bytes_to_keep) const {
        if (gpu()) {
            #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
            cuda::memory::Pool(cuda::Device(id(), true)).trim(bytes_to_keep);
            #else
            (void) bytes_to_keep;
            return;
            #endif
        }
    }

    inline Device Device::current(Type type) {
        if (type == Type::CPU)
            return Device{};
        #ifdef NOA_ENABLE_CUDA
        return Device(Type::GPU, cuda::Device::current().id(), true);
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    inline void Device::current(Device device) {
        if (device.gpu()) {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device::current(cuda::Device(device.id(), true));
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    inline size_t Device::count(Type type) {
        if (type == Type::CPU) {
            return 1;
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::Device::count();
            #else
            return 0;
            #endif
        }
    }

    inline bool Device::any(Type type) {
        return Device::count(type) != 0;
    }

    inline std::vector<Device> Device::all(Type type) {
        if (type == Type::CPU) {
            return {Device(Type::CPU)};
        } else {
            #ifdef NOA_ENABLE_CUDA
            std::vector<Device> devices;
            size_t count = cuda::Device::count();
            devices.reserve(count);
            for (auto id: irange(static_cast<int>(count)))
                devices.emplace_back(Type::GPU, id, true);
            return devices;
            #else
            return {};
            #endif
        }
    }

    inline Device Device::mostFree(Type type) {
        if (type == Type::CPU) {
            return {Device(Type::CPU)};
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device most_free = cuda::Device::mostFree();
            return Device(Type::GPU, most_free.id(), true);
            #else
            NOA_THROW("GPU backend is not detected");
            #endif
        }
    }

    inline void Device::validate_(Type type, int id) {
        if (type == Type::CPU) {
            if (id != -1)
                NOA_THROW("The device ID for the CPU should be -1, but got {}", id);
            return;
        } else if (type == Type::GPU) {
            if (id < 0)
                NOA_THROW("GPU device ID should be positive, but got {}", id);

            #ifdef NOA_ENABLE_CUDA
            const size_t count = cuda::Device::count();
            if (static_cast<size_t>(id) + 1 > count)
                NOA_THROW("CUDA device ID {} does not match any of CUDA device(s) detected (count:{})", id, count);
            #else
            NOA_THROW("GPU backend is not detected");
            #endif
        } else {
            NOA_THROW("Unrecognized type. Should be {} or {}, but got {}", Type::CPU, Type::GPU, type);
        }
    }
}
