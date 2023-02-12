#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::utils {
    // Returns the pointer attributes of ptr.
    NOA_IH cudaPointerAttributes pointer_attributes(const void* ptr) {
        cudaPointerAttributes attr{};
        NOA_THROW_IF(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    }

    // If ptr can be accessed by the device, returns ptr or the corresponding device pointer.
    // If ptr cannot be accessed by the device, returns nullptr.
    template<typename T>
    NOA_IH T* device_pointer(T* ptr, Device device) {
        const auto attr = pointer_attributes(ptr);
        if (attr.type == cudaMemoryTypeUnregistered)
            return nullptr;
        else if (attr.type == cudaMemoryTypeDevice)
            return device.get() == attr.device ? ptr : nullptr;
        else if (attr.type == cudaMemoryTypeHost)
            return static_cast<T*>(attr.devicePointer);
        else if (attr.type == cudaMemoryTypeManaged)
            return ptr;
        return nullptr; // unreachable
    }
    #define NOA_ASSERT_DEVICE_PTR(ptr, device) NOA_ASSERT(::noa::cuda::utils::device_pointer(ptr, device) != nullptr)
    #define NOA_ASSERT_DEVICE_OR_NULL_PTR(ptr, device) NOA_ASSERT(ptr == nullptr || ::noa::cuda::utils::device_pointer(ptr, device) != nullptr)

    // If ptr can be accessed by the host, returns ptr. Otherwise, returns nullptr.
    template<typename T>
    NOA_IH T* host_pointer(T* ptr) {
        const auto attr = pointer_attributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    // Aligned array that generates vectorized load/store in CUDA.
    // It seems that the maximum load size is 16 bytes.
    template<typename T, i64 VECTOR_SIZE>
    struct alignas(sizeof(T) * VECTOR_SIZE) AlignedVector {
        T data[VECTOR_SIZE];
    };

    // Returns the number of T elements that can be vectorized to one load/store call.
    // CUDA vectorized load/store stops at 16bytes/128bits, so return early if the type cannot
    // be vectorized, so it can be abstracted away.
    template<typename T>
    NOA_IHD constexpr i64 max_vector_count(const T* pointer) {
        if constexpr (!noa::math::is_power_of_2(sizeof(T))) {
            return 1;
        } else {
            constexpr std::uintptr_t vec2_alignment = alignof(AlignedVector<T, 2>);
            constexpr std::uintptr_t vec4_alignment = alignof(AlignedVector<T, 4>);
            constexpr std::uintptr_t vec8_alignment = alignof(AlignedVector<T, 8>);
            constexpr std::uintptr_t vec16_alignment = alignof(AlignedVector<T, 16>);
            const auto address = reinterpret_cast<std::uintptr_t>(pointer);

            if constexpr (sizeof(T) == 16) {
                return 1;
            } else if constexpr (sizeof(T) == 8) {
                if (address % vec2_alignment == 0)
                    return 2;
                else
                    return 1;
            } else if constexpr (sizeof(T) == 4) {
                if (address % vec4_alignment == 0)
                    return 4;
                else if (address % vec2_alignment == 0)
                    return 2;
                else
                    return 1;
            } else if constexpr (sizeof(T) == 2) {
                if (address % vec8_alignment == 0)
                    return 8;
                else if (address % vec4_alignment == 0)
                    return 4;
                else if (address % vec2_alignment == 0)
                    return 2;
                else
                    return 1;
            } else {
                if (address % vec16_alignment == 0)
                    return 16;
                else if (address % vec8_alignment == 0)
                    return 8;
                else if (address % vec4_alignment == 0)
                    return 4;
                else if (address % vec2_alignment == 0)
                    return 2;
                else
                    return 1;
            }
        }
    }
}
