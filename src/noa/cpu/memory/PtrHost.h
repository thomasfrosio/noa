/// \file noa/cpu/memory/PtrHost.h
/// \brief Hold memory on the host.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include <fftw3.h> // fftw_malloc

#include <string>
#include <type_traits>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/traits/BaseTypes.h"

// Alignment
// =========
//
//  - By specifying the alignment requirement:
//      Allocating and returning an aligned pointer is very easy to do with malloc(), with for examples
//      https://stackoverflow.com/questions/227897. With C++17, alignment_of() is probably a better idea these
//      alternatives because we can use free() on the aligned pointer. In C++17, overloads of the "new" operator can
//      extract the alignment requirement of the type (probably using something like alignof or alignment_of, which
//      is known at compile time) and call the overload of new that takes in the alignment requirement. The underlying
//      operations are then probably similar to aligned_alloc().
//
//      The issue with this is that we cannot use calloc() and we might implement our own aligned calloc(). The only
//      issue with this is that we then need to use our own free() functions (which internally calls the normal free()
//      function provided by C). Note that using calloc() can have real advantages: https://stackoverflow.com/a/2688522
//
//      Note: X-BYTE aligned means that the leading BYTE address needs to be a multiple of X, X being a power of 2.
//      Note: malloc() is supposed to return a pointer that is aligned to is sufficiently well aligned for any of the
//            basic types (long, long double, pointers, etc.) up to the type with the maximum possible alignment, i.e.
//            std::max_align_t. With more specialized things (e.g. SIMD), this might not be enough and over-alignment
//            might be necessary.
//
//  - FFTW:
//      In the common case where we are using a SIMD-using FFTW, we should guarantee proper alignment for SIMD.
//      As such, all PtrHost<T> data, when T is float/double or cfloat_t/cdouble_t, should be allocated/freed using the
//      PtrHost::alloc/dealloc static functions. These functions will call FFTW to do the allocation. Ultimately,
//      we could do the allocation ourselves, but we would need to know the alignment required by FFTW, which might
//      not be straightforward.
//
// - Conclusion:
//      Specifying the type to the "allocator", and therefore using new (or fftw_malloc) as opposed to malloc/
//      aligned_alloc(), is easier for us since they support any type alignment requirement (at least in C++17).
//      TODO In the future, we should replace fftw_malloc/fftw_free by aligned_alloc().
//           If we decide to add aligned_calloc() support, we'll have to add aligned_free() as well.
//
// PtrHost
// =======
//
// The goal is not to replace unique_ptr or shared_ptr, since they offer functionalities that PtrHost does not, but
// to keep track of the number of managed elements and offers a container-like API.

namespace noa::cpu::memory {
    /// Manages a host pointer.
    template<typename T>
    class PtrHost {
    public:
        struct Deleter {
            void operator()(T* ptr) noexcept {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t> ||
                              std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                    fftw_free(ptr);
                } else {
                    delete[] ptr;
                }
            }
        };

        /// Allocates \p n elements of type \p T on the \b host.
        /// \note If \p T is a float, double, cfloat_t or cdouble_t, it uses fftw_malloc/fftw_free, which ensures that
        ///       the returned pointer has the necessary alignment (by calling memalign or its equivalent) for the
        ///       SIMD-using FFTW to use SIMD instructions.
        static unique_t<T[], Deleter> alloc(size_t elements) {
            T* out;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t>) {
                out = static_cast<T*>(fftwf_malloc(elements * sizeof(T)));
            } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                out = static_cast<T*>(fftw_malloc(elements * sizeof(T)));
            } else {
                out = new(std::nothrow) T[elements];
            }
            if (!out)
                NOA_THROW("Failed to allocate {} {} on the heap", elements, string::human<T>());
            return {out, Deleter{}};
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
