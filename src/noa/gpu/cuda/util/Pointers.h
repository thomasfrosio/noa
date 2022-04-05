/// \file noa/gpu/cuda/memory/Pointers.h
/// \brief Pointer attributes and utilities for CUDA.
/// \author Thomas - ffyr2w
/// \date 2 Feb 2022
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Traits.h"

namespace noa::cuda::util {
    /// Returns the pointer attributes of \p ptr.
    NOA_IH cudaPointerAttributes getAttributes(const void* ptr) {
        cudaPointerAttributes attr{};
        NOA_THROW_IF(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    }

    /// If \p ptr can be accessed by the device, returns \p ptr or the corresponding device pointer.
    /// If \p ptr cannot be accessed by the device, returns nullptr.
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

    /// If \p ptr can be accessed by the host, returns \p ptr. Otherwise, returns nullptr.
    template<typename T>
    NOA_IH T* hostPointer(T* ptr) {
        const auto attr = getAttributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    /// Checks if \p ptr is accessible by the stream's device. If so, return \p ptr (or for pinned memory, return the
    /// corresponding device pointer). Otherwise, allocates new memory asynchronously, copy \p ptr to that new memory
    /// and return a pointer to that new memory.
    template<typename T>
    NOA_IH shared_t<const T[]> ensureDeviceAccess(const shared_t<const T[]>& ptr, Stream& stream, size_t elements) {
        const T* tmp = devicePointer(ptr.get(), stream.device());
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

    /// Returns the number of \p T elements that can be vectorized to one load/store call. Can be 1, 2 or 4.
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

    /// Simple wrapper to add, programmatically, __restrict__ attributes to pointers.
    template<bool RESTRICT, typename Pointer>
    struct accessor_t {
    public:
        using ptr_type = std::conditional_t<RESTRICT, Pointer __restrict__, Pointer>;
        using value_type = std::remove_pointer<Pointer>;
        NOA_DEVICE explicit accessor_t(Pointer ptr) : m_data(ptr) {};
        NOA_DEVICE ptr_type get() noexcept { return m_data; }
        NOA_DEVICE auto& operator[](size_t i) noexcept { return m_data[i]; }
    private:
        ptr_type m_data;
    };

    namespace traits {
        template<typename T> struct p_is_accessor : std::false_type {};
        template<typename T, bool R> struct p_is_accessor<accessor_t<R, T>> : std::true_type {};
        template<typename T> using is_accessor = std::bool_constant<p_is_accessor<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> inline constexpr bool is_accessor_v = is_accessor<T>::value;
    }
}
