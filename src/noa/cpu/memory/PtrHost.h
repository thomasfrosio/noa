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
    /// Manages a host pointer.
    template<typename T>
    class PtrHost {
    public:
        static constexpr size_t ALIGNMENT = traits::is_float_v<T> || traits::is_complex_v<T> ? 128 :
                                            traits::is_int_v<T> ? 64 :
                                            alignof(T);
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

        /// Allocates some elements of uninitialized storage. Throws if the allocation fails.
        static unique_t<T[], Deleter> alloc(size_t elements) {
            if (!elements)
                return {};
            T* out;
            if constexpr(traits::is_data_v<T>)
                out = static_cast<T*>(std::aligned_alloc(ALIGNMENT, elements * sizeof(T)));
            else
                out = new(std::nothrow) T[elements];
            if (!out)
                NOA_THROW("Failed to allocate {} {} on the heap", elements, string::human<T>());
            return {out, Deleter{}};
        }

        /// Allocates some elements, all initialized to 0. Throws if the allocation fails.
        static unique_t<T[], DeleterCalloc> calloc(size_t elements) {
            using namespace ::noa::traits;
            if (!elements)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            const size_t offset = ALIGNMENT - 1 + sizeof(void*);

            void* calloc_ptr = std::calloc(elements * sizeof(T) + offset, 1);
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
        explicit PtrHost(size_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

    public:
        /// Returns the host pointer.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// Attach the lifetime of the managed object with an \p alias.
        /// \details Constructs a shared_ptr which shares ownership information with the managed object,
        ///          but holds an unrelated and unmanaged pointer \p alias. If the returned shared_ptr is
        ///          the last of the group to go out of scope, it will call the stored deleter for the
        ///          managed object of this instance. However, calling get() on this shared_ptr will always
        ///          return a copy of \p alias. It is the responsibility of the programmer to make sure that
        ///          \p alias remains valid as long as the managed object exists.
        template<typename U>
        [[nodiscard]] constexpr std::shared_ptr<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        constexpr T* begin() noexcept { return m_ptr.get(); }
        constexpr const T* begin() const noexcept { return m_ptr.get(); }

        /// Returns a pointer pointing at the last + 1 element of the managed data.
        constexpr T* end() noexcept { return m_ptr.get() + m_elements; }
        constexpr const T* end() const noexcept { return m_ptr.get() + m_elements; }

        constexpr std::reverse_iterator<T> rbegin() noexcept { return m_ptr.get(); }
        constexpr std::reverse_iterator<const T> rbegin() const noexcept { return m_ptr.get(); }
        constexpr std::reverse_iterator<T> rend() noexcept { return m_ptr.get() + m_elements; }
        constexpr std::reverse_iterator<const T> rend() const noexcept { return m_ptr.get() + m_elements; }

        /// Returns a reference at index \p idx. There's no bound check.
        constexpr T& operator[](size_t idx) { return m_ptr.get()[idx]; }
        constexpr const T& operator[](size_t idx) const { return m_ptr.get()[idx]; }

        /// Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(noa::traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{};
        size_t m_elements{0};
    };
}
