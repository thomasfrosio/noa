#pragma once

#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cuda/Device.hpp"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Runtime.hpp"

namespace noa::cuda {
    /// Returns the pointer attributes of ptr.
    inline auto pointer_attributes(const void* ptr) -> cudaPointerAttributes {
        cudaPointerAttributes attr;
        check(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    }

    /// If ptr can be accessed by the device, returns ptr or the corresponding device pointer.
    /// If ptr cannot be accessed by the device, returns nullptr.
    template<typename T>
    auto device_pointer(T* ptr, Device device) -> T* {
        const auto attr = pointer_attributes(ptr);
        switch (attr.type) {
            case cudaMemoryTypeUnregistered:
                return nullptr;
            case cudaMemoryTypeDevice:
                return device.get() == attr.device ? ptr : nullptr;
            case cudaMemoryTypeHost:
                return static_cast<T*>(attr.devicePointer);
            case cudaMemoryTypeManaged:
                return ptr;
        }
        return nullptr; // unreachable
    }

    /// If ptr can be accessed by the host, returns ptr. Otherwise, returns nullptr.
    template<typename T>
    auto host_pointer(T* ptr) -> T* {
        const auto attr = pointer_attributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    /// Returns the address of constant_pointer on the device.
    template<typename T = void>
    auto constant_to_device_pointer(const void* constant_pointer) -> T* {
        void* device_pointer;
        check(cudaGetSymbolAddress(&device_pointer, constant_pointer));
        return static_cast<T*>(device_pointer);
    }
}
