#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Exception.hpp"
#include "noa/core/Types.hpp"

namespace noa::cpu {
    template<typename T>
    struct AllocatorHeapDeleter {
        void operator()(T* ptr) noexcept {
            if constexpr(nt::is_numeric_v<T>)
                std::free(ptr);
            else
                delete[] ptr;
        }
    };

    template<typename T>
    struct AllocatorHeapDeleterCalloc {
        void operator()(T* ptr) noexcept {
            std::free(reinterpret_cast<void**>(ptr)[-1]);
        }
    };

    // Allocates memory from the heap.
    // Also provides memory-aligned alloc and calloc functions.
    // Floating-points and complex floating-points are aligned to 128 bytes.
    // Integral types are aligned to 64 bytes.
    // Anything else uses the type's alignment as reported by alignof().
    template<typename Value>
    class AllocatorHeap {
    public:
        static_assert(!std::is_pointer_v<Value> and !std::is_reference_v<Value> and !std::is_const_v<Value>);
        static constexpr size_t ALIGNMENT =
                nt::is_real_or_complex_v<Value> ? 128 :
                nt::is_int_v<Value> ? 64 : alignof(Value);

        using value_type = Value;
        using shared_type = std::shared_ptr<value_type[]>;
        using alloc_deleter_type = AllocatorHeapDeleter<value_type>;
        using calloc_deleter_type = AllocatorHeapDeleterCalloc<value_type>;
        using alloc_unique_type = std::unique_ptr<value_type[], alloc_deleter_type>;
        using calloc_unique_type = std::unique_ptr<value_type[], calloc_deleter_type>;

    public:
        // Allocates some elements of uninitialized storage. Throws if the allocation fails.
        static alloc_unique_type allocate(i64 elements) {
            if (elements <= 0)
                return {};
            value_type* out;
            if constexpr (nt::is_numeric_v<value_type>) {
                out = static_cast<value_type*>(std::aligned_alloc(
                        ALIGNMENT, static_cast<size_t>(elements) * sizeof(value_type)));
            } else {
                out = new(std::nothrow) value_type[static_cast<size_t>(elements)];
            }
            check(out, "Failed to allocate {} {} on the heap", elements, ns::to_human_readable<value_type>());
            return {out, alloc_deleter_type{}};
        }

        // Allocates some elements, all initialized to 0. Throws if the allocation fails.
        static calloc_unique_type calloc(i64 elements) {
            if (elements <= 0)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            const size_t offset = ALIGNMENT - 1 + sizeof(void*);

            void* calloc_ptr = std::calloc(static_cast<size_t>(elements) * sizeof(value_type) + offset, 1);
            check(calloc_ptr, "Failed to allocate {} {} on the heap", elements, ns::to_human_readable<value_type>());

            // Align to the requested value, leaving room for the original calloc value.
            void* aligned_ptr = reinterpret_cast<void*>(((uintptr_t)calloc_ptr + offset) & ~(ALIGNMENT - 1));
            reinterpret_cast<void**>(aligned_ptr)[-1] = calloc_ptr;
            return {static_cast<value_type*>(aligned_ptr), calloc_deleter_type{}};
        }
    };
}
#endif
