#ifndef NOA_UNIFIED_DEVICE_
#error "Implementation header"
#endif

namespace noa {
    NOA_IH constexpr Device::Device(Device::Type type, int id, bool unsafe) {
        m_id = type == Type::CPU ? -1 : id;
        if (!unsafe)
            validate_(m_id);
    }

    NOA_IH Device::Device(std::string_view device, bool unsafe) {
        try {
            m_id = parse_(device);
            if (!unsafe)
                validate_(m_id);
        } catch (...) {
            NOA_THROW("Failed to parse the input ID \"{}\" into a valid ID", device);
        }
    }

    NOA_IH void Device::synchronize() const {
        #ifdef NOA_ENABLE_CUDA
        if (this->gpu())
            cuda::Device(this->id(), true).synchronize();
        #endif
    }

    NOA_IH void Device::reset() const {
        #ifdef NOA_ENABLE_CUDA
        if (this->gpu())
            cuda::Device(this->id(), true).reset();
        #endif
    }

    NOA_IH std::string Device::summary() const {
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

    NOA_IH DeviceMemory Device::memory() const {
        if (cpu()) {
            #ifdef NOA_ENABLE_CPU
            const cpu::DeviceMemory mem_info = cpu::Device::memory();
            return {mem_info.total, mem_info.free};
            #else
            return {};
            #endif
        } else {
            #ifdef NOA_ENABLE_CUDA
            const cuda::DeviceMemory mem_info = cuda::Device(this->id(), true).memory();
            return {mem_info.total, mem_info.free};
            #else
            return {};
            #endif
        }
    }

    NOA_IH size_t Device::count(Type type) {
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

    /// Whether there's any device available of this type.
    NOA_IH bool Device::any(Type type) {
        return Device::count(type) != 0;
    }

    NOA_IH std::vector<Device> Device::all(Type type) {
        if (type == Type::CPU) {
            return {Device{Type::CPU}};
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

    NOA_IH Device Device::mostFree(Type type) {
        if (type == Type::CPU) {
            return {Device{Type::CPU}};
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::Device most_free = cuda::Device::mostFree();
            return Device(Type::GPU, most_free.id(), true);
            #else
            NOA_THROW("");
            #endif
        }
    }

    NOA_IH int Device::parse_(std::string_view str) {
        str = string::trim(string::lower(str));
        const size_t length = str.length();

        if (string::startsWith(str, "cpu")) {
            if (length == 3)
                return -1;

        } else if (string::startsWith(str, "gpu")) {
            if (length == 3)
                return 0;
            else if (length >= 5 && str[3] == ':')
                return string::toInt<int>(std::string{str.data() + 4});

        } else if (string::startsWith(str, "cuda")) {
            if (length == 4)
                return 0;
            else if (length >= 6 && str[4] == ':')
                return string::toInt<int>(std::string{str.data() + 5});
        }
        NOA_THROW("Device type not recognized");
    }

    NOA_IH void Device::validate_(int id) {
        if (id == -1)
            return;
        if (id < 0)
            NOA_THROW("Device IDs should be positive, got {}", id);

        #ifdef NOA_ENABLE_CUDA
        const size_t count = cuda::Device::count();
        if (static_cast<size_t>(id) + 1 > count)
            NOA_THROW("CUDA device ID {} does not match any of CUDA device(s) detected (count:{})", id, count);
        #else
        NOA_THROW("No GPU backend detected but a GPU device ID ({}) was entered", id);
        #endif
    }
}
