#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Exception.hpp"
#include "noa/gpu/cuda/Device.hpp"
#include "noa/gpu/cuda/Types.hpp"

#if defined(NOA_IS_OFFLINE)
namespace noa::cuda {
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
#define NOA_ASSERT_DEVICE_PTR(ptr, device) NOA_ASSERT(::noa::cuda::device_pointer(ptr, device) != nullptr)
#define NOA_ASSERT_DEVICE_OR_NULL_PTR(ptr, device) NOA_ASSERT(ptr == nullptr || ::noa::cuda::device_pointer(ptr, device) != nullptr)

    // If ptr can be accessed by the host, returns ptr. Otherwise, returns nullptr.
    template<typename T>
    NOA_IH T* host_pointer(T* ptr) {
        const auto attr = pointer_attributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    // Returns the address of constant_pointer on the device.
    template<typename T = void>
    NOA_IH T* constant_to_device_pointer(const void* constant_pointer) {
        void* device_pointer;
        NOA_THROW_IF(cudaGetSymbolAddress(&device_pointer, constant_pointer));
        return static_cast<T*>(device_pointer);
    }

    // Collects and stores kernel argument addresses.
    // This implies that the arguments should stay valid during the lifetime of this object.
    template<typename... Args>
    class CollectArgumentAddresses {
    public:
        // The array of pointers is initialized with the arguments (which are not copied).
        // Const-cast is required since CUDA excepts void**, not const void**.
        explicit CollectArgumentAddresses(Args&& ... args)
                : m_pointers{const_cast<void*>(static_cast<const void*>(&args))...} {}

        [[nodiscard]] void** pointers() const { return static_cast<void**>(m_pointers); }

    private:
        void* m_pointers[std::max(size_t{1}, sizeof...(Args))]{}; // non-empty
    };
}
#endif

namespace noa::cuda {
    // Aligned array that generates vectorized load/store in CUDA.
    template<typename T, size_t VECTOR_SIZE>
    struct alignas(sizeof(T) * VECTOR_SIZE) AlignedVector {
        T data[VECTOR_SIZE];
    };

    // Returns the number of T elements that can be vectorized to one load/store call.
    // CUDA vectorized load/store stops at 16bytes/128bits, so return early if the type cannot
    // be vectorized, so it can be abstracted away.
    template<typename T>
    NOA_IHD constexpr i64 max_vector_count(const T* pointer) {
        if constexpr (!is_power_of_2(sizeof(T))) {
            return 1;
        } else {
            constexpr auto vec2_alignment = alignof(AlignedVector<T, 2>);
            constexpr auto vec4_alignment = alignof(AlignedVector<T, 4>);
            constexpr auto vec8_alignment = alignof(AlignedVector<T, 8>);
            constexpr auto vec16_alignment = alignof(AlignedVector<T, 16>);
            const auto address = reinterpret_cast<decltype(vec2_alignment)>(pointer); // T* to uintptr

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
