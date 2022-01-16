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
    /// Manages a host pointer. This object is not copyable.
    /// \tparam T   T of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type, which
    ///             is basically any type excluding a reference/array or const type.
    /// \throw      If an error occurs when data is allocated or freed.
    template<typename T>
    class PtrHost {
    public:
        /// Allocates \p n elements of type \p T on the \b host.
        /// \note If \p T is a float, double, cfloat_t or cdouble_t, it uses fftw_malloc/fftw_free, which ensures that
        ///       the returned pointer has the necessary alignment (by calling memalign or its equivalent) for the
        ///       SIMD-using FFTW to use SIMD instructions.
        static NOA_HOST T* alloc(size_t n) {
            T* out;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t>) {
                out = static_cast<T*>(fftwf_malloc(n * sizeof(T)));
            } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                out = static_cast<T*>(fftw_malloc(n * sizeof(T)));
            } else {
                out = new(std::nothrow) T[n];
            }
            if (!out)
                NOA_THROW("Failed to allocate {} {} on the heap", n, string::typeName<T>());
            return out;
        }

        /// De-allocates \p data.
        /// \note \p data should have been allocated with PtrHost::alloc().
        static NOA_HOST void dealloc(T* data) noexcept {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t> ||
                          std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                fftw_free(data);
            } else {
                delete[] data;
            }
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrHost() = default;

        /// Allocates \p elements elements of type \p T on the heap.
        /// \param elements This is attached to the underlying managed pointer and is fixed for the entire
        ///                 life of the object. Use elements() to access it. The number of allocated bytes is
        ///                 (at least) equal to `elements * sizeof(T)`, see bytes().
        ///
        /// \note   The created instance is the owner of the data.
        ///         To get a non-owning pointer, use get().
        ///         To release the ownership, use release().
        NOA_HOST explicit PtrHost(size_t elements) : m_elements(elements), m_ptr(alloc(m_elements)) {}

        /// Creates an instance from existing data.
        /// \param[in] ptr  Host pointer to hold on.
        ///                 If it is a nullptr, \p elements should be 0.
        ///                 If it is not a nullptr, it should correspond to \p elements.
        /// \param elements Number of \p T elements in \p ptr.
        NOA_HOST PtrHost(T* ptr, size_t elements) noexcept
                : m_elements(elements), m_ptr(ptr) {}

        /// Move constructor. \p to_move is not meant to be used after this call.
        NOA_HOST PtrHost(PtrHost<T>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \p to_move is not meant to be used after this call.
        NOA_HOST PtrHost<T>& operator=(PtrHost<T>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrHost(const PtrHost<T>& to_copy) = delete;
        PtrHost<T>& operator=(const PtrHost<T>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr T* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr T* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* data() const noexcept { return m_ptr; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        NOA_HOST constexpr T* begin() noexcept { return m_ptr; }
        NOA_HOST constexpr const T* begin() const noexcept { return m_ptr; }

        /// Returns a pointer pointing at the last + 1 element of the managed data.
        NOA_HOST constexpr T* end() noexcept { return m_ptr + m_elements; }
        NOA_HOST constexpr const T* end() const noexcept { return m_ptr + m_elements; }

        NOA_HOST constexpr std::reverse_iterator<T> rbegin() noexcept { return m_ptr; }
        NOA_HOST constexpr std::reverse_iterator<const T> rbegin() const noexcept { return m_ptr; }
        NOA_HOST constexpr std::reverse_iterator<T> rend() noexcept { return m_ptr + m_elements; }
        NOA_HOST constexpr std::reverse_iterator<const T> rend() const noexcept { return m_ptr + m_elements; }

        /// Returns a reference at index \p idx. There's no bound check.
        NOA_HOST constexpr T& operator[](size_t idx) { return m_ptr[idx]; }
        NOA_HOST constexpr const T& operator[](size_t idx) const { return m_ptr[idx]; }

        /// Clears the underlying data, if necessary. empty() will evaluate to true after this call.
        NOA_HOST void reset() {
            dealloc(m_ptr);
            m_elements = 0;
            m_ptr = nullptr;
        }

        /// Clears the underlying data, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size_t elements) {
            dealloc(m_ptr);
            m_elements = elements;
            m_ptr = alloc(m_elements);
        }

        /// Resets the underlying data.
        /// \param[in] data     Host pointer to hold on. If it is not a nullptr, it should correspond to \p elements.
        /// \param elements     Number of \p T elements in \p data.
        NOA_HOST void reset(T* data, size_t elements) {
            dealloc(m_ptr);
            m_elements = elements;
            m_ptr = data;
        }

        /// Releases the ownership of the managed pointer, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call and empty() returns true.
        [[nodiscard]] NOA_HOST T* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates the data.
        NOA_HOST ~PtrHost() noexcept { dealloc(m_ptr); }

    private:
        size_t m_elements{0};
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<T>, T*> m_ptr{nullptr};
    };
}
