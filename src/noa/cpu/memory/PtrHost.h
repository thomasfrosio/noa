/// \file noa/cpu/memory/PtrHost.h
/// \brief Hold memory on the host.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include <string>
#include <type_traits>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa::cpu::memory {
    /// Allocates and manages a heap-allocated memory region.
    /// Also provides memory-aligned alloc and calloc functions.
    /// \note Floating-points and complex floating-points are aligned to 128 bytes.
    ///       Integral types are aligned to 64 bytes.
    ///       Anything else uses the type's alignment as reported by alignof().
    template<typename T>
    class PtrHost {
    public:
        struct Deleter {
            void operator()(T* ptr) noexcept {
                if constexpr(traits::is_data_v<T>)
                    std::free(ptr);
                else
                    delete[] ptr;
            }
        };

        struct DeleterCalloc {
            void operator()(T* ptr) noexcept {
                std::free(reinterpret_cast<void**>(ptr)[-1]);
            }
        };

    public:
        using alloc_unique_t = unique_t<T[], Deleter>;
        using calloc_unique_t = unique_t<T[], DeleterCalloc>;
        static constexpr size_t ALIGNMENT = traits::is_float_v<T> || traits::is_complex_v<T> ?
                                            128 : traits::is_int_v<T> ? 64 : alignof(T);

    public:
        /// Allocates some elements of uninitialized storage. Throws if the allocation fails.
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        static alloc_unique_t alloc(I elements) {
            if (elements <= 0)
                return {};
            T* out;
            if constexpr (traits::is_data_v<T>)
                out = static_cast<T*>(std::aligned_alloc(ALIGNMENT, static_cast<size_t>(elements) * sizeof(T)));
            else
                out = new(std::nothrow) T[elements];
            if (!out)
                NOA_THROW("Failed to allocate {} {} on the heap", elements, string::human<T>());
            return {out, Deleter{}};
        }

        /// Allocates some elements, all initialized to 0. Throws if the allocation fails.
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        static calloc_unique_t calloc(I elements) {
            if (elements <= 0)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            const size_t offset = ALIGNMENT - 1 + sizeof(void*);

            void* calloc_ptr = std::calloc(static_cast<size_t>(elements) * sizeof(T) + offset, 1);
            if (!calloc_ptr)
                NOA_THROW("Failed to allocate {} {} on the heap", elements, string::human<T>());

            // Align to the requested value, leaving room for the original calloc value.
            void* aligned_ptr = reinterpret_cast<void*>(((uintptr_t)calloc_ptr + offset) & ~(ALIGNMENT - 1));
            reinterpret_cast<void**>(aligned_ptr)[-1] = calloc_ptr;
            return {static_cast<T*>(aligned_ptr), DeleterCalloc{}};
        }

    public:
        /// Creates an empty instance. Use one of the operator assignment to allocate new data.
        PtrHost() = default;
        constexpr /*implicit*/ PtrHost(std::nullptr_t) {}

        /// Allocates \p elements elements of type \p T on the heap.
        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        explicit PtrHost(I elements) : m_ptr(alloc(elements)), m_elements(static_cast<size_t>(elements)) {
            NOA_ASSERT(elements >= 0);
        }

    public: // Getters
        /// Returns the host pointer.
        [[nodiscard]] constexpr T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// Attach the lifetime of the managed object with \p alias.
        /// \details Constructs a shared_ptr which shares ownership information with the managed object,
        ///          but holds an unrelated and unmanaged pointer \p alias. If the returned shared_ptr is
        ///          the last of the group to go out of scope, it will call the stored deleter for the
        ///          managed object of this instance. However, calling get() on this shared_ptr will always
        ///          return a copy of \p alias. It is the responsibility of the programmer to make sure that
        ///          \p alias remains valid as long as the managed object exists. This functions performs no
        ///          heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename U>
        [[nodiscard]] constexpr std::shared_ptr<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        /// Returns the shape of the allocated data as a row vector.
        [[nodiscard]] constexpr size4_t shape() const noexcept { return {1, 1, 1, m_elements}; }

        /// Returns the strides of the allocated data as a C-contiguous row vector.
        [[nodiscard]] constexpr size4_t strides() const noexcept { return shape().strides(); }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a View of the allocated data as a C-contiguous row vector.
        template<typename I>
        [[nodiscard]] constexpr View<T, I> view() const noexcept { return {m_ptr.get(), shape(), strides()}; }

    public: // Iterators
        [[nodiscard]] constexpr T* begin() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* end() const noexcept { return m_ptr.get() + m_elements; }

        [[nodiscard]] constexpr T& front() const noexcept { return *begin(); }
        [[nodiscard]] constexpr T& back() const noexcept { return *(end() - 1); }

    public: // Accessors
        /// Returns a reference at index \p idx. There's no bound check.
        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] constexpr T& operator[](I idx) const { return m_ptr.get()[idx]; }

        /// Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{};
        size_t m_elements{0};
    };
}
