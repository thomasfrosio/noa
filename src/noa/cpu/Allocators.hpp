#pragma once

#include <memory>
#include "noa/core/Error.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::cpu {
    /// Allocates memory from the heap.
    /// \details This allocator is designed to allocate medium to large memory regions, as it enforces an alignment
    ///          of 256 bytes by default (or more if the type requires it). Moreover, the allocation is done with
    ///          malloc-like functions, thus returns uninitialized memory regions, i.e. it is undefined behavior to
    ///          directly read from this memory. Similarly, the allocator's deleter is simply freeing the memory,
    ///          thereby requiring T to be trivially destructible.
    class AllocatorHeap {
    public:
        struct Deleter {
            size_t size{};
            void operator()(void* ptr) const noexcept {
                std::free(ptr);
                m_bytes_currently_allocated -= size;
            }
        };

        struct DeleterCalloc {
            size_t size{};
            void operator()(void* ptr) const noexcept {
                std::free(static_cast<void**>(ptr)[-1]);
                m_bytes_currently_allocated -= size;
            }
        };

        template<typename T> using allocate_type = std::unique_ptr<T[], Deleter>;
        template<typename T> using calloc_type = std::unique_ptr<T[], DeleterCalloc>;

    public:
        /// Allocates some elements of uninitialized storage. Throws if the allocation fails.
        template<nt::allocatable_type T, size_t MIN_ALIGNMENT = 256>
        static auto allocate(i64 n_elements) -> allocate_type<T> {
            if (n_elements <= 0)
                return {};

            // TODO aligned_alloc requires the size to be a multiple of the alignment.
            //      We could use the same strategy as with calloc, but for now this is fine.
            //      https://en.cppreference.com/w/c/memory/aligned_alloc
            constexpr size_t alignment = std::max(alignof(T), MIN_ALIGNMENT);
            const size_t n_bytes = next_multiple_of(static_cast<size_t>(n_elements) * sizeof(T), alignment);
            auto out = static_cast<T*>(std::aligned_alloc(alignment, n_bytes));
            check(out, "Failed to allocate {} {} on the heap", n_elements, ns::stringify<T>());
            m_bytes_currently_allocated += n_bytes;
            return {out, Deleter{n_bytes}};
        }

        /// Allocates some elements, with the underlying bytes initialized to 0. Throws if the allocation fails.
        template<nt::allocatable_type T, size_t ALIGNMENT = 256>
        static auto calloc(i64 n_elements) -> calloc_type<T> {
            if (n_elements <= 0)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            constexpr size_t alignment = std::max(alignof(T), ALIGNMENT);
            const size_t offset = alignment - 1 + sizeof(void*);
            const size_t n_bytes = static_cast<size_t>(n_elements) * sizeof(T) + offset;
            void* calloc_ptr = std::calloc(n_bytes, 1);
            check(calloc_ptr, "Failed to allocate {} {} on the heap", n_elements, ns::stringify<T>());
            m_bytes_currently_allocated += n_bytes;

            // Align to the requested value, leaving room for the original calloc value.
            auto aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(calloc_ptr) + offset) & ~(alignment - 1));
            static_cast<void**>(aligned_ptr)[-1] = calloc_ptr;
            return {static_cast<T*>(aligned_ptr), DeleterCalloc{n_bytes}};
        }

        [[nodiscard]] static auto bytes_currently_allocated() -> size_t { return m_bytes_currently_allocated.load(); }

    private:
        inline static std::atomic<size_t> m_bytes_currently_allocated{};
    };
}
