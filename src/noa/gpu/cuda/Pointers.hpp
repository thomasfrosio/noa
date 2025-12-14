#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Device.hpp"
#include "noa/gpu/cuda/Error.hpp"
#include "noa/gpu/cuda/Runtime.hpp"

namespace noa::cuda {
    /// Aligned array that generates vectorized load/store in CUDA.
    /// TODO Replace with Vec<T, VECTOR_SIZE, sizeof(T) * VECTOR_SIZE>?
    template<typename T, usize VECTOR_SIZE>
    struct alignas(sizeof(T) * VECTOR_SIZE) AlignedVector {
        T data[VECTOR_SIZE];
    };

    /// Aligned array used to generate vectorized load/store in CUDA.
    template<typename T, usize N, usize A>
    struct alignas(A) AlignedBuffer {
        using value_type = T;
        constexpr static usize SIZE = N;
        T data[N];
    };

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

    /// Collects and stores kernel argument addresses.
    /// This implies that the arguments should stay valid during the lifetime of this object.
    template<typename... Args>
    class CollectArgumentAddresses {
    public:
        /// The array of pointers is initialized with the arguments (which are not copied).
        /// Const-cast is required since CUDA expects void**.
        explicit CollectArgumentAddresses(Args&& ... args) :
            m_pointers{const_cast<void*>(static_cast<const void*>(&args))...} {}

        [[nodiscard]] auto pointers() -> void** { return static_cast<void**>(m_pointers); }

    private:
        void* m_pointers[max(usize{1}, sizeof...(Args))]{};
    };

    /// Returns the minimum address alignment for the given accessors.
    /// \details The vectorization happens along the width, so vectorization is turned off if any of the
    ///          width stride is not 1. This function checks that the alignment is preserved at the beginning
    ///          of every row. The block size and number of elements per thread is assumed to be a power of two
    ///          (in which case if the rows are aligned, the beginning of every block will be too).
    /// \param accessors    Tuple of 4d accessors, as this is intended for the *ewise core functions.
    ///                     AccessorValue is supported and preserves the vector size. Passing an empty
    ///                     tuple returns the maximum alignment (for global memory word-count), 16-byte.
    /// \param shape_bdh    BDH shape. Empty dimensions do not affect the alignment, so if certain
    ///                     dimensions are known to be contiguous, the dimension size can be set to 1
    ///                     to skip it.
    template<typename T, typename Index>
    requires (nt::tuple_of_accessor_nd<T, 4> or nt::empty_tuple<T>)
    constexpr auto min_address_alignment(
        const T& accessors,
        const Shape<Index, 3>& shape_bdh
    ) -> usize {
        auto get_alignment = [](const void* pointer) -> usize{
            // Global memory instructions support reading or
            // writing words of size equal to 1, 2, 4, 8, or 16 bytes.
            const auto address = reinterpret_cast<uintptr_t>(pointer);
            if (is_multiple_of(address, 16))
                return 16;
            if (is_multiple_of(address, 8))
                return 8;
            if (is_multiple_of(address, 4))
                return 4;
            if (is_multiple_of(address, 2))
                return 2;
            return 1;
        };

        usize alignment = 16;
        accessors.for_each([&]<typename U>(const U& accessor) {
            if constexpr (nt::accessor_pure<U>) {
                if (accessor.stride(3) == 1) {
                    usize i_alignment = get_alignment(accessor.get());
                    const auto strides = accessor.strides().template as_safe<usize>();

                    // Make sure every row is aligned to the current alignment.
                    // If not, try to decrease the alignment until reaching the minimum
                    // alignment for this type.
                    constexpr auto SIZE = sizeof(typename U::value_type);
                    for (; i_alignment >= 2; i_alignment /= 2) {
                        if ((shape_bdh[2] == 1 or is_multiple_of(strides[2] * SIZE, i_alignment)) and
                            (shape_bdh[1] == 1 or is_multiple_of(strides[1] * SIZE, i_alignment)) and
                            (shape_bdh[0] == 1 or is_multiple_of(strides[0] * SIZE, i_alignment)))
                            break;
                    }
                    alignment = min(alignment, i_alignment);
                } else {
                    // Since the vectorization is set up at compile time, we have no choice but
                    // to turn off the vectorization for everyone if one accessor is strided.
                    alignment = 1;
                }
            }
        });
        return alignment;
    }

    /// Computes the maximum vector size allowed for the given inputs/outputs.
    template<usize ALIGNMENT, typename... T>
    consteval auto maximum_allowed_aligned_buffer_size() -> usize {
        usize size{ALIGNMENT};
        auto get_size = [&]<typename V>() -> usize {
            using value_t = nt::mutable_value_type_t<V>;
            if constexpr (nt::accessor_value<V>) {
                return size; // AccessorValue shouldn't affect the vector size
            } else if constexpr (nt::accessor_pure<V> and is_power_of_2(sizeof(value_t))) {
                constexpr usize RATIO = sizeof(value_t) / alignof(value_t); // non naturally aligned types
                constexpr usize N = (ALIGNMENT / alignof(value_t)) / RATIO;
                return max(usize{1}, N); // clamp to one for cases where ALIGNMENT < alignof(value_t)
            } else {
                static_assert(nt::accessor_pure<V>);
                // If size is not a power of two, memory accesses cannot fully coalesce;
                // there's no point in increasing the word count.
                return 1;
            }
        };

        // To fully coalesce and to ensure that threads work on the same elements,
        // we have to use the same vector size for all inputs/outputs.
        constexpr auto accessors = (nt::type_list_t<T>{} + ...);
        [&]<typename... U>(nt::TypeList<U...>) {
            ((size = min(size, get_size.template operator()<U>())), ...);
        }(accessors);
        return size;
    }

    template<typename T, usize ALIGNMENT, usize N>
    struct to_aligned_buffer {
        template<typename U>
        static constexpr auto get_type() {
            using value_t = nt::mutable_value_type_t<U>;
            if constexpr (nt::accessor_pure<U> and is_power_of_2(sizeof(value_t))) {
                constexpr usize RATIO = sizeof(value_t) / alignof(value_t); // non naturally aligned types
                constexpr usize AA = min(alignof(value_t) * N * RATIO, ALIGNMENT);
                constexpr usize A = max(AA, alignof(value_t));
                return std::type_identity<AlignedBuffer<value_t, N, A>>{};
            } else {
                return std::type_identity<AlignedBuffer<value_t, N, alignof(value_t)>>{};
            }
        }

        template<typename... U>
        static constexpr auto get(nt::TypeList<U...>) {
            return std::type_identity<Tuple<typename decltype(get_type<U>())::type...>>{};
        }

        using type = decltype(get(nt::type_list_t<T>{}))::type;
    };
    template<typename T, usize ALIGNMENT, usize N>
    using to_aligned_buffer_t = to_aligned_buffer<T, ALIGNMENT, N>::type;

    /// Whether the aligned buffers are actually over-aligned compared to the original type.
    /// In other words, whether using vectorized loads/stores is useful.
    /// This is used to fall back on a non-vectorized implementation at compile time,
    /// thus reducing the number of kernels that need to be generated.
    template<typename... T>
    constexpr auto is_vectorized() -> usize {
        constexpr auto aligned_buffers = (nt::type_list_t<T>{} + ...);
        if constexpr (nt::empty_tuple<T...>) {
            return false; // no inputs and no outputs
        } else {
            return []<typename... U>(nt::TypeList<U...>) {
                return ((alignof(U) > alignof(typename U::value_type)) or ...);
            }(aligned_buffers);
        }
    }
}
