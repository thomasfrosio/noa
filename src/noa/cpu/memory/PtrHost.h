/**
 * @file noa/cpu/memory/PtrHost.h
 * @brief Simple pointer holding memory on the host.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <fftw3.h> // fftw_malloc

#include <string>
#include <type_traits>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/traits/BaseTypes.h"

/*
 * Data structure alignment
 * ========================
 *
 *  - By specifying the alignment requirement:
 *      See https://stackoverflow.com/questions/227897 to do this with malloc(). Nowadays, alignment_of() is probably
 *      a better idea than malloc.
 *      Note: X-BYTE aligned means that the leading BYTE address needs to be a multiple of X, X being a power of 2.
 *      Note: malloc() is supposed to return a pointer that is aligned to std::max_align_t. This is sufficiently well
 *            aligned for any of the basic types (long, long double, pointers, etc.). With more specialized things,
 *            this might not be enough and over-alignment might be necessary.
 *
 *  - By specifying the type:
 *      In C++17, overloads of the "new" operator can extract the alignment requirement of the type (probably using
 *      something like alignof or alignment_of, which is at compile time) and call the overload that takes in the
 *      alignment. The underlying operations are then probably similar to aligned_alloc().
 *
 *  - FFTW:
 *      In the common case where we are using a SIMD-using FFTW, we can guarantee proper alignment for SIMD. As such,
 *      all PtrHost<T> data, when T is float/double or cfloat_t/cdouble_t, should be allocated/freed using the
 *      PtrHost::alloc/dealloc static functions. This is done automatically.
 *
 * - Conclusion:
 *      Specifying the type to the "allocator", and therefore using new (or fftw_malloc) as opposed to malloc/
 *      aligned_alloc(), is easier for my use case and support any type alignment requirement (at least in C++17).
 *
 * Smart pointers?
 * ===============
 *
 * The goal is not to replace unique_ptr or shared_ptr, since they offer functionalities that PtrHost does not, but
 * PtrHost keeps track of the number of managed elements and offers a container-like API.
 */

namespace Noa::Memory {
    /**
     * Manages a host pointer. This object cannot be used on the device and is not copyable.
     * @tparam Type     Type of the underlying pointer. Anything allowed by @c Traits::is_valid_ptr_type.
     * @throw           @c Noa::Exception, if an error occurs when data is allocated or freed.
     */
    template<typename Type>
    class PtrHost {
    private:
        size_t m_elements{0};
        std::enable_if_t<Noa::Traits::is_valid_ptr_type_v<Type>, Type*> m_ptr{nullptr};

    public:
        /**
         * Allocates @a n elements of type @a T.
         * @note If @a T is a float/cfloat_t or double/cdouble_t, it uses fftw_malloc/fftw_free, which ensures that
         *       the returned pointer has the necessary alignment (by calling memalign or its equivalent) for the
         *       SIMD-using FFTW to use SIMD instructions.
         */
        template<typename T>
        NOA_HOST T* alloc(size_t n) {
            T* out;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t> ||
                          std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                out = static_cast<T*>(fftw_malloc(n * sizeof(T)));
            } else {
                out = new(std::nothrow) Type[n];
            }
            if (!out)
                NOA_THROW("Failed to allocate {} elements of type {} on the heap", n, String::typeName<Type>());
            return out;
        }

        /// De-allocates @a data. @warning @a data should have been allocated with PtrHost::alloc.
        template<typename T>
        NOA_HOST void dealloc(T* data) noexcept {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, cfloat_t> ||
                          std::is_same_v<T, double> || std::is_same_v<T, cdouble_t>) {
                fftw_free(data);
            } else {
                delete[] data;
            }
        }

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrHost() = default;

        /**
         * Allocates @a elements elements of type @a Type on the heap.
         * @param elements  This is attached to the underlying managed pointer and is fixed for the entire
         *                  life of the object. Use elements() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrHost(size_t elements) : m_elements(elements) { alloc_(); }

        /**
         * Creates an instance from existing data.
         * @param[in] ptr   Host pointer to hold on.
         *                  If it is a nullptr, @a elements should be 0.
         *                  If it is not a nullptr, it should correspond to @a elements.
         * @param elements  Number of @a Type elements in @a ptr.
         */
        NOA_HOST PtrHost(Type* ptr, size_t elements) noexcept
                : m_elements(elements), m_ptr(ptr) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrHost(PtrHost<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrHost<Type>& operator=(PtrHost<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy functions.
        PtrHost(const PtrHost<Type>& to_copy) = delete;
        PtrHost<Type>& operator=(const PtrHost<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_ptr; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }

        /** How many bytes are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /** Returns a pointer pointing at the beginning of the managed data. */
        NOA_HOST constexpr Type* begin() noexcept { return m_ptr; }
        NOA_HOST constexpr const Type* begin() const noexcept { return m_ptr; }

        /** Returns a pointer pointing at the end + 1 of the managed data. */
        NOA_HOST constexpr Type* end() noexcept { return m_ptr + m_elements; }
        NOA_HOST constexpr const Type* end() const noexcept { return m_ptr + m_elements; }

        NOA_HOST constexpr std::reverse_iterator<Type> rbegin() noexcept { return m_ptr; }
        NOA_HOST constexpr std::reverse_iterator<const Type> rbegin() const noexcept { return m_ptr; }
        NOA_HOST constexpr std::reverse_iterator<Type> rend() noexcept { return m_ptr + m_elements; }
        NOA_HOST constexpr std::reverse_iterator<const Type> rend() const noexcept { return m_ptr + m_elements; }

        /** Returns a reference at index @a idx. There's no bound check. */
        NOA_HOST constexpr Type& operator[](size_t idx) { return *(m_ptr + idx); }
        NOA_HOST constexpr const Type& operator[](size_t idx) const { return *(m_ptr + idx); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
            m_ptr = nullptr;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size_t elements) {
            dealloc_();
            m_elements = elements;
            alloc_();
        }

        /**
         * Resets the underlying data.
         * @param[in] data  Host pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param elements  Number of @a Type elements in @a data.
         */
        NOA_HOST void reset(Type* data, size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
        }

        /**
         * Releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call and empty() returns true.
         */
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /** Deallocates the data. */
        ~PtrHost() noexcept { dealloc_(); }

    private:
        NOA_HOST void alloc_() { m_ptr = alloc<Type>(m_elements); }
        NOA_HOST void dealloc_() noexcept { dealloc(m_ptr); }
    };
}
