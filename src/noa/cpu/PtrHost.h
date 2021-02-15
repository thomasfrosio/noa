/**
 * @file noa/cpu/PtrHost.h
 * @brief Simple pointer (for arithmetic types) holding memory on the host.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <string>
#include <type_traits>
#include <utility>      // std::exchange
#include <cstddef>      // size_t
#include <cstdlib>      // malloc, free
#include <cstring>      // std::memcpy

#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/string/Format.h"     // String::format
#include "noa/util/traits/BaseTypes.h"  // Traits::is_complex_v

/* Data structure alignment:
 *  - By specifying the alignment requirement:
 *      See https://stackoverflow.com/questions/227897 to do this with malloc(). Nowadays, alignment_of() is probably
 *      a better idea than malloc.
 *      Note: X-byte aligned means that the leading byte address needs to be a multiple of X, X being a power of 2.
 *      Note: malloc() is supposed to return a pointer that is aligned to std::max_align_t. This is sufficiently well
 *            aligned for any of the basic types (long, long double, pointers, etc.). With more specialized things,
 *            this might not be enough and over-alignment might be necessary.
 *
 *  - By specifying the type:
 *      In C++17, overloads of the "new" operator can extract the alignment requirement of the type (probably using
 *      something like alignof or alignment_of, which is at compile time) and call the overload that takes in the
 *      alignment. The underlying operations are then probably similar to aligned_alloc().
 *
 * - Conclusion:
 *      Specifying the type to the "allocator", and therefore using new as opposed to malloc/aligned_alloc(), is easier
 *      for my use case and support any type alignment requirement (at least in C++17).
 *
 *  Thomas (ffyr2w), 11 Jan 2021.
 */

namespace Noa {
    /**
     * Holds a pointer pointing to some "arithmetic" type, usually used as a fixed-sized dynamic array.
     * Ownership:   Data can be owned, and ownership can be switched on and off at any time.
     *              Ownership implies that the destructor will delete the pointer.
     *
     * @tparam Type Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw       @c Noa::Exception, if an error occurs when the data is allocated or freed.
     */
    template<typename Type>
    class PtrHost {
    private:
        size_t m_elements{0};
        std::enable_if_t<Noa::Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_ptr{nullptr};

    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrHost() = default;

        /**
         * Allocates @a elements elements of type @a Type on the heap.
         * @param elements  This is attached to the underlying managed pointer and is fixed for the entire
         *                  life of the object. Use elements() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrHost(size_t elements) : m_elements(elements) { alloc_(elements); }

        /** Allocates @a elements elements of type @a Type on the heap and value initialize them with @a value. */
        NOA_HOST explicit PtrHost(size_t elements, Type value) : m_elements(elements) {
            alloc_(elements, value);
        }

        /**
         * Creates an instance from a existing data.
         * @param elements  Number of @a Type elements in @a dev_ptr.
         * @param[in] ptr   Device pointer to hold on.
         *                  If it is a nullptr, @a elements should be 0.
         *                  If it is not a nullptr, it should correspond to @a elements.
         * @param owner     Whether or not this new instance should own @a ptr.
         */
        NOA_HOST PtrHost(size_t elements, Type* ptr, bool owner) noexcept:
                m_elements(elements), m_ptr(ptr), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrHost(const PtrHost<Type>& to_copy) noexcept
                : m_elements(to_copy.m_elements), m_ptr(to_copy.m_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrHost(PtrHost<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrHost<Type>& operator=(const PtrHost<Type>& to_copy) = delete;
        PtrHost<Type>& operator=(PtrHost<Type>&& to_move) = delete;

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
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return m_ptr; }

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
        NOA_HOST constexpr Type& operator[](size_t idx) noexcept { return *(m_ptr + idx); }
        NOA_HOST constexpr const Type& operator[](size_t idx) const noexcept { return *(m_ptr + idx); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size_t elements) {
            dealloc_();
            m_elements = elements;
            alloc_(m_elements);
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param elements  Number of @a Type elements in @a data.
         * @param[in] data  Host pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner     Whether or not this new instance should own @a data.
         */
        NOA_HOST void reset(size_t elements, Type* data, bool owner) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /** Returns a human-readable description of the PtrHost. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: host, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, static_cast<void*>(m_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~PtrHost() { dealloc_(); }

    private:
        // Allocates. Otherwise, throw.
        NOA_HOST void alloc_(size_t elements) {
            m_ptr = new(std::nothrow) Type[elements];
            if (!m_ptr)
                NOA_THROW("failed to allocate {} bytes on the heap", elements * sizeof(Type));
        }

        // Allocates and value initialize. Otherwise, throw.
        NOA_HOST void alloc_(size_t elements, Type value) {
            m_ptr = new(std::nothrow) Type[elements]{value};
            if (!m_ptr)
                NOA_THROW("failed to allocate {} bytes on the heap", elements * sizeof(Type));
        }

        // Deallocates the underlying data, if any and if the instance is the owner.
        NOA_HOST void dealloc_() {
            if (!m_ptr)
                return;
            if (is_owner)
                delete[] m_ptr;
            else
                m_ptr = nullptr;
        }
    };

    template<class T>
    [[nodiscard]] NOA_IH std::string toString(PtrHost<T> ptr) { return ptr.toString(); }
}

template<typename T>
struct fmt::formatter<Noa::PtrHost<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::PtrHost<T>& ptr, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::toString(ptr), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::PtrHost<T>& stream) {
    os << Noa::toString(stream);
    return os;
}

