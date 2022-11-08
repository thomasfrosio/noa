#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/Traits.h"

namespace noa::cuda::utils {
    // Returns the pointer attributes of ptr.
    NOA_IH cudaPointerAttributes getAttributes(const void* ptr) {
        cudaPointerAttributes attr{};
        NOA_THROW_IF(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    }

    // If ptr can be accessed by the device, returns ptr or the corresponding device pointer.
    // If ptr cannot be accessed by the device, returns nullptr.
    template<typename T>
    NOA_IH T* devicePointer(T* ptr, Device device) {
        const auto attr = getAttributes(ptr);
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
    #define NOA_ASSERT_DEVICE_PTR(ptr, device) NOA_ASSERT(::noa::cuda::utils::devicePointer(ptr, device) != nullptr)

    // If ptr can be accessed by the host, returns ptr. Otherwise, returns nullptr.
    template<typename T>
    NOA_IH T* hostPointer(T* ptr) {
        const auto attr = getAttributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    // Checks if ptr is accessible by the stream's device. If so, return ptr (or for pinned memory, return the
    // corresponding device pointer). Otherwise, allocates new memory asynchronously, copy ptr to that new memory
    // and return a pointer to that new memory.
    template<typename T>
    NOA_IH shared_t<T[]> ensureDeviceAccess(const shared_t<T[]>& ptr, Stream& stream, dim_t elements) {
        T* tmp = devicePointer(ptr.get(), stream.device());
        if (!tmp) {
            shared_t<T[]> buffer = memory::PtrDevice<T>::alloc(elements, stream);
            NOA_THROW_IF(cudaMemcpyAsync(buffer.get(), ptr.get(), elements * sizeof(T),
                                         cudaMemcpyDefault, stream.id()));
            stream.attach(ptr, buffer);
            return buffer;
        } else {
            return {ptr, tmp}; // pinned memory can have a different device ptr, so alias tmp
        }
    }

    // Checks if ptr is accessible by the stream's device. If so, return ptr (or for pinned memory, return the
    // corresponding device pointer). Otherwise, allocates new memory asynchronously, copy ptr to that new memory
    // and return a pointer to that new memory.
    template<typename T, typename U, typename = std::enable_if_t<std::is_same_v<noa::traits::remove_ref_cv_t<T>, U>>>
    NOA_IH T* ensureDeviceAccess(T* ptr, Stream& stream, memory::PtrDevice<U>& allocator, dim_t elements) {
        T* tmp = devicePointer(ptr, stream.device());
        if (!tmp) {
            allocator = memory::PtrDevice<U>{elements, stream};
            NOA_THROW_IF(cudaMemcpyAsync(allocator.get(), ptr, allocator.elements() * sizeof(T),
                                         cudaMemcpyDefault, stream.id()));
            return allocator.get();
        } else {
            return tmp;
        }
    }

    // Returns the number of T elements that can be vectorized to one load/store call. Can be 1, 2 or 4.
    template<typename T>
    NOA_IHD uint maxVectorCount(const T* pointer) {
        const auto address = reinterpret_cast<uint64_t>(pointer);
        constexpr int vec2_alignment = alignof(traits::aligned_vector_t<T, 2>);
        constexpr int vec4_alignment = alignof(traits::aligned_vector_t<T, 4>);
        if (address % vec4_alignment == 0)
            return 4;
        else if (address % vec2_alignment == 0)
            return 2;
        else
            return 1;
    }
}
