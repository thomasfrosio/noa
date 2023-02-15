#ifndef NOA_UNIFIED_DEVICE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

namespace noa {
    inline constexpr Device::Device(DeviceType type, i32 id)
            : m_id(type == DeviceType::CPU ? -1 : id) {
        validate_(type, m_id);
    }

    inline Device::Device(std::string_view name) {
        const auto[type, id] = parse_type_and_id_(name);
        validate_(type, id);
        m_id = id;
    }

    inline constexpr Device::Device(DeviceType type, i32 id, DeviceUnchecked)
            : m_id(type == DeviceType::CPU ? -1 : id) {}

    inline Device::Device(std::string_view name, DeviceUnchecked) {
        const auto[type, id] = parse_type_and_id_(name);
        m_id = id;
    }

    inline std::string Device::summary() const {
        if (is_cpu()) {
            return cpu::Device::summary();
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::Device(this->id(), cuda::Device::DeviceUnchecked{}).summary();
            #else
            return {};
            #endif
        }
    }

    inline DeviceMemory Device::memory_capacity() const {
        if (is_cpu()) {
            const cpu::DeviceMemory mem_info = cpu::Device::memory();
            return {mem_info.total, mem_info.free};
        } else {
            #ifdef NOA_ENABLE_CUDA
            const cuda::DeviceMemory mem_info = cuda::Device(this->id(), cuda::Device::DeviceUnchecked{}).memory();
            return {mem_info.total, mem_info.free};
            #else
            return {};
            #endif
        }
    }

    inline void Device::set_cache_threshold(size_t threshold_bytes) const {
        if (is_gpu()) {
            #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
            cuda::memory::Pool(cuda::Device(id(), cuda::Device::DeviceUnchecked{})).set_threshold(threshold_bytes);
            #else
            (void) threshold_bytes;
            return;
            #endif
        }
    }

    inline void Device::trim_cache(size_t bytes_to_keep) const {
        if (is_gpu()) {
            #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11020
            cuda::memory::Pool(cuda::Device(id(), cuda::Device::DeviceUnchecked{})).trim(bytes_to_keep);
            #else
            (void) bytes_to_keep;
            return;
            #endif
        }
    }

    inline Device Device::current(DeviceType type) {
        if (type == DeviceType::CPU)
            return Device{};
        #ifdef NOA_ENABLE_CUDA
        return Device(DeviceType::GPU, cuda::Device::current().id(), DeviceUnchecked{});
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    inline void Device::set_current(Device device) {
        if (device.is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device::set_current(cuda::Device(device.id(), cuda::Device::DeviceUnchecked{}));
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    inline i32 Device::count(DeviceType type) {
        if (type == DeviceType::CPU) {
            return 1;
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::Device::count();
            #else
            return 0;
            #endif
        }
    }

    inline bool Device::any(DeviceType type) {
        return Device::count(type) != 0;
    }

    inline std::vector<Device> Device::all(DeviceType type) {
        if (type == DeviceType::CPU) {
            return {Device(DeviceType::CPU)};
        } else {
            #ifdef NOA_ENABLE_CUDA
            std::vector<Device> devices;
            const i32 count = cuda::Device::count();
            devices.reserve(static_cast<size_t>(count));
            for (auto id: irange(count))
                devices.emplace_back(DeviceType::GPU, id, DeviceUnchecked{});
            return devices;
            #else
            return {};
            #endif
        }
    }

    inline Device Device::most_free(DeviceType type) {
        if (type == DeviceType::CPU) {
            return Device(DeviceType::CPU);
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device most_free = cuda::Device::most_free();
            return Device(DeviceType::GPU, most_free.id(), DeviceUnchecked{});
            #else
            NOA_THROW("GPU backend is not detected");
            #endif
        }
    }

    inline std::pair<DeviceType, i32> Device::parse_type_and_id_(std::string_view name) {
        std::string str_ = string::lower(string::trim(name));
        const size_t length = name.length();

        i32 id{};
        DeviceType type;
        i32 error{0};
        if (string::starts_with(str_, "cpu")) {
            if (length == 3) {
                id = -1;
                type = DeviceType::CPU;
            } else {
                NOA_THROW("CPU device name \"{}\" is not supported", str_);
            }

        } else if (string::starts_with(str_, "gpu")) {
            if (length == 3) {
                id = 0;
            } else if (length >= 5 && str_[3] == ':') {
                id = string::parse<i32>(std::string(str_.data() + 4), error);
            } else {
                NOA_THROW("GPU device name \"{}\" is not supported", str_);
            }
            type = DeviceType::GPU;

        } else if (string::starts_with(str_, "cuda")) {
            if (length == 4) {
                id = 0;
            } else if (length >= 6 && str_[4] == ':') {
                id = string::parse<i32>(std::string(str_.data() + 5), error);
            } else {
                NOA_THROW("CUDA device name \"{}\" is not supported", str_);
            }
            type = DeviceType::GPU;
        } else {
            NOA_THROW("Failed to parse \"{}\" as a valid device name", str_);
        }

        if (error)
            NOA_THROW("Failed to parse the device ID. {}", string::parse_error_message<i32>(str_, error));
        return {type, id};
    }

    inline void Device::validate_(DeviceType type, i32 id) {
        if (type == DeviceType::CPU) {
            if (id != -1)
                NOA_THROW("The device ID for the CPU should be -1, but got {}", id);
            return;
        } else if (type == DeviceType::GPU) {
            if (id < 0)
                NOA_THROW("GPU device ID should be positive, but got {}", id);

            #ifdef NOA_ENABLE_CUDA
            const i32 count = cuda::Device::count();
            if (id + 1 > count)
                NOA_THROW("CUDA device ID {} does not match any of CUDA device(s) detected (count:{})", id, count);
            #else
            NOA_THROW("GPU backend is not detected");
            #endif
        } else {
            NOA_THROW("DEV: Missing type");
        }
    }
}
