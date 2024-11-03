#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Exception.hpp"
#include "noa/core/Types.hpp"

namespace noa::cpu {
    template<typename T>
    struct AllocatorHeapDeleter {
        void operator()(T* ptr) noexcept {
            std::free(ptr);
        }
    };

    template<typename T>
    struct AllocatorHeapDeleterCalloc {
        void operator()(T* ptr) noexcept {
            std::free(reinterpret_cast<void**>(ptr)[-1]);
        }
    };

    /// Allocates memory from the heap.
    /// \details This allocator is designed to allocate medium to large memory regions, as it enforces an alignment
    ///          of 256 bytes by default (or more if the type requires it). Moreover, the allocation is done with
    ///          malloc-like functions, thus returns uninitialized memory regions, i.e. it is undefined behavior to
    ///          directly read from this memory. Similarly, the allocator's deleter is simply freeing the memory,
    ///          thereby requiring T to be trivially destructible.
    template<typename T>
    class AllocatorHeap {
    public:
        static_assert(not std::is_pointer_v<T> and
                      not std::is_reference_v<T> and
                      not std::is_const_v<T> and
                      std::is_trivially_destructible_v<T>);

        using value_type = T;
        using shared_type = std::shared_ptr<value_type[]>;
        using alloc_deleter_type = AllocatorHeapDeleter<value_type>;
        using calloc_deleter_type = AllocatorHeapDeleterCalloc<value_type>;
        using alloc_unique_type = std::unique_ptr<value_type[], alloc_deleter_type>;
        using calloc_unique_type = std::unique_ptr<value_type[], calloc_deleter_type>;
        static constexpr size_t SIZEOF = sizeof(value_type);
        static constexpr size_t ALIGNOF = alignof(value_type);

    public:
        /// Allocates some elements of uninitialized storage. Throws if the allocation fails.
        template<size_t ALIGNMENT = 256>
        static alloc_unique_type allocate(i64 n_elements) {
            if (n_elements <= 0)
                return {};

            constexpr size_t alignment = std::max(ALIGNOF, ALIGNMENT);
            auto out = static_cast<value_type*>(std::aligned_alloc(alignment, static_cast<size_t>(n_elements) * SIZEOF));
            check(out, "Failed to allocate {} {} on the heap", n_elements, ns::stringify<value_type>());
            return {out, alloc_deleter_type{}};
        }

        /// Allocates some elements, with the underlying bytes initialized to 0. Throws if the allocation fails.
        template<size_t ALIGNMENT = 256>
        static calloc_unique_type calloc(i64 n_elements) {
            if (n_elements <= 0)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            constexpr size_t alignment = std::max(ALIGNOF, ALIGNMENT);
            const size_t offset = alignment - 1 + sizeof(void*);

            void* calloc_ptr = std::calloc(static_cast<size_t>(n_elements) * SIZEOF + offset, 1);
            check(calloc_ptr, "Failed to allocate {} {} on the heap", n_elements, ns::stringify<value_type>());

            // Align to the requested value, leaving room for the original calloc value.
            void* aligned_ptr = reinterpret_cast<void*>(((uintptr_t)calloc_ptr + offset) & ~(alignment - 1));
            reinterpret_cast<void**>(aligned_ptr)[-1] = calloc_ptr;
            return {static_cast<value_type*>(aligned_ptr), calloc_deleter_type{}};
        }
    };
}
#endif
