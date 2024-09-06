#pragma once

#include "noa/core/Config.hpp"

namespace noa::cuda {
    /// Aligned array that generates vectorized load/store in CUDA.
    /// TODO Replace with Vec<T, VECTOR_SIZE, sizeof(T) * VECTOR_SIZE>?
    template<typename T, size_t VECTOR_SIZE>
    struct alignas(sizeof(T) * VECTOR_SIZE) AlignedVector {
        T data[VECTOR_SIZE];
    };
}

#ifdef NOA_IS_OFFLINE
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Device.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Runtime.hpp"

namespace noa::cuda {
    /// Returns the pointer attributes of ptr.
    inline cudaPointerAttributes pointer_attributes(const void* ptr) {
        cudaPointerAttributes attr;
        check(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    }

    /// If ptr can be accessed by the device, returns ptr or the corresponding device pointer.
    /// If ptr cannot be accessed by the device, returns nullptr.
    template<typename T>
    T* device_pointer(T* ptr, Device device) {
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
    T* host_pointer(T* ptr) {
        const auto attr = pointer_attributes(ptr);
        return attr.type == cudaMemoryTypeDevice ? nullptr : ptr;
    }

    /// Returns the address of constant_pointer on the device.
    template<typename T = void>
    T* constant_to_device_pointer(const void* constant_pointer) {
        void* device_pointer;
        check(cudaGetSymbolAddress(&device_pointer, constant_pointer));
        return static_cast<T*>(device_pointer);
    }

    /// Collects and stores kernel argument addresses.
    /// This implies that the arguments should stay valid during the lifetime of this object.
    template<typename... Args>
    class CollectArgumentAddresses {
    public:
        /// The array of pointers is initialized with the arguments (which are not copied).
        /// Const-cast is required since CUDA expects void**.
        explicit CollectArgumentAddresses(Args&& ... args) :
            m_pointers{const_cast<void*>(static_cast<const void*>(&args))...} {}

        [[nodiscard]] void** pointers() { return static_cast<void**>(m_pointers); }

    private:
        void* m_pointers[max(size_t{1}, sizeof...(Args))]{};
    };

    /// Returns the number of T elements that can be vectorized to one load/store call.
    /// CUDA vectorized load/store stops at 16bytes/128bits, so return early if the type cannot
    /// be vectorized, in the hope that it will help the compiler optimize the vectorized kernel
    /// instantiation away.
    template<typename T>
    constexpr i64 max_vector_count(const T* pointer) {
        if constexpr (not is_power_of_2(sizeof(T))) {
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
                return 1;
            } else if constexpr (sizeof(T) == 4) {
                if (address % vec4_alignment == 0)
                    return 4;
                if (address % vec2_alignment == 0)
                    return 2;
                return 1;
            } else if constexpr (sizeof(T) == 2) {
                if (address % vec8_alignment == 0)
                    return 8;
                if (address % vec4_alignment == 0)
                    return 4;
                if (address % vec2_alignment == 0)
                    return 2;
                return 1;
            } else {
                if (address % vec16_alignment == 0)
                    return 16;
                if (address % vec8_alignment == 0)
                    return 8;
                if (address % vec4_alignment == 0)
                    return 4;
                if (address % vec2_alignment == 0)
                    return 2;
                return 1;
            }
        }
    }

    /// Returns the maximum vector size (used for vectorized load/store) for the given accessors.
    /// \details The vectorization happens along the width, so vectorization is turned off if any of the
    ///          width stride is not 1. This function checks that the alignment is preserved at the beginning
    ///          of every block work size and at the beginning of every row.
    /// \param tuple_of_4d_accessors    Tuple of 4d accessors, as this is intended for the *ewise core functions.
    ///                                 AccessorValue is supported and preserves the vector size. It's probably a good
    ///                                 idea to turn off the vectorization if every accessor is an AccessorValue, but
    ///                                 this function cannot do it, since the vectorization may depend on other
    ///                                 accessors.
    /// \param n_elements_per_thread_x  The number of elements per thread along the width. The vector size will not
    ///                                 exceed this value. If it is not a power of 2, the vectorization is turned
    ///                                 off.
    /// \param block_size_x             Number of threads per block assigned to process the row(s).
    /// \param shape_bdh                BDH shape. Empty dimensions do not affect the alignment and thus the
    ///                                 vectorization. For instance, cases where the inputs are all contiguous,
    ///                                 all dimensions can be set to 1.
    template<nt::tuple_of_accessor_nd<4> T, typename Index>
    constexpr auto maximum_vector_size(
        const T& tuple_of_4d_accessors,
        u32 n_elements_per_thread_x,
        u32 block_size_x,
        const Shape3<Index>& shape_bdh
    ) -> u32 {
        if (n_elements_per_thread_x == 1 or not is_power_of_2(n_elements_per_thread_x))
            return 1;

        u32 vector_size = n_elements_per_thread_x; // maximum vector size
        tuple_of_4d_accessors.for_each([&](const auto& accessor) {
            if constexpr (nt::accessor_pure<decltype(accessor)>) {
                if (accessor.stride(3) == 1) {
                    const auto strides = accessor.strides().template as<u32>();
                    auto i_vector_size = static_cast<u32>(max_vector_count(accessor.get()));

                    // If the alignment at the start of the batch/block is not enough,
                    // decrease the vector size. Repeat until vector
                    for (; i_vector_size >= 2; i_vector_size /= 2) {
                        // Make sure all blocks are aligned (block_size_x is usually a multiple of 32,
                        // so it is usually okay).
                        const auto block_work_size_x = block_size_x * max(n_elements_per_thread_x, i_vector_size);
                        if (is_multiple_of(block_work_size_x, i_vector_size) and
                            (shape_bdh[2] == 1 or is_multiple_of(strides[2], i_vector_size)) and
                            (shape_bdh[1] == 1 or is_multiple_of(strides[1], i_vector_size)) and
                            (shape_bdh[0] == 1 or is_multiple_of(strides[0], i_vector_size)))
                            break;
                    }
                    vector_size = min(vector_size, i_vector_size);
                } else {
                    vector_size = 1; // turn off vectorization
                }
            }
        });

        // min may help the compiler see that vector_size <= n_elements_per_thread_x.
        return min(vector_size, n_elements_per_thread_x);
    }
}
#endif
