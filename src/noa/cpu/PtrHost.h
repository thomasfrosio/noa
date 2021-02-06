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

#include "noa/util/Types.h"
#include "noa/util/string/Format.h"     // String::format
#include "noa/util/traits/BaseTypes.h"  // Traits::is_complex_v

/* Data structure alignment:
 *  - By specifying the alignment requirement:
 *      See https://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library
 *      to do this with malloc(). Nowadays, alignment_of() is probably a better idea than malloc.
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
     *              Ownership implies:
     *                  1) the destructor will delete the pointer.
     *                  2) the copy constructor will perform a deep copy.
     *
     * @tparam Type Type of the underlying pointer. Should be an non-const arithmetic such as defined
     *              by std::is_arithmetic or a non-cost std::complex<float|double|long double>.
     *              Arithmetics are all aligned to std::max_align_t (see discussion above), so the type
     *              is mostly here to add some type safety.
     *
     * @warning     The copy constructor or copy assignment operator of an owning PtrHost<> will copy
     *              the underlying data.
     */
    template<typename Type>
    class PtrHost {
    private:
        size_t m_size{};
        std::enable_if_t<(std::is_arith_v<Type> || Traits::is_complex_v<Type>) &&
                         !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> &&
                         !std::is_const_v<Type>, Type*> m_ptr{nullptr};

    public:
        // Whether or not the underlying pointer is owned by this instance.
        // Can be changed at any time, since it only affects the copy ctor/assignment operator and the dtor.
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        PtrHost() = default;

        /**
         * Allocates @a size elements of type @a Type on the heap.
         * @param[in] size      This is fixed for the life of the object. Use size() to access it.
         *                      The number of bytes allocated is (at least) equal to `size * sizeof(Type)`.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         *
         * @warning The allocation may fail and the underlying data can be a nullptr. As such, new
         *          instances should be checked, by using the bool operator or get().
         */
        explicit PtrHost(size_t size) noexcept: m_size(size), m_ptr(alloc_(size)) {}

        /**
         * Creates an instance from an existing pointer.
         * @param[in] size      Number of @a Type elements in @a ptr.
         * @param[in] ptr       PtrHost to hold on. If it is not a nullptr, it should correspond to @a size.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        PtrHost(size_t size, Type* ptr, bool own_ptr = false) noexcept : m_size(size), m_ptr(ptr), is_owner(own_ptr) {}

        /**
         * Copy constructor.
         * @note    If @a ptr owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy. In this case, the new instance
         *          will not own the data.
         */
        PtrHost(const PtrHost<Type>& ptr) noexcept : m_size(ptr.m_size), is_owner(ptr.is_owner) {
            m_ptr = (is_owner && ptr.m_ptr) ? copy_(ptr.m_ptr) : ptr.m_ptr;
        }

        /**
         * Move constructor.
         * @note    @a ptr is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        PtrHost(PtrHost<Type>&& ptr) noexcept
                : m_size(ptr.m_size), m_ptr(std::exchange(ptr.m_ptr, nullptr)), is_owner(ptr.is_owner) {}

        /** Copy assignment operator. */
        inline constexpr PtrHost<Type>& operator=(const PtrHost<Type>& ptr) noexcept {
            m_size = ptr.m_size;
            m_ptr = (is_owner && ptr.m_ptr) ? copy_(ptr.m_ptr) : ptr.m_ptr;
            is_owner = ptr.is_owner;
            return *this;
        }

        /** Move assignment operator. */
        inline constexpr PtrHost<Type>& operator=(PtrHost<Type>&& ptr) noexcept {
            m_size = ptr.m_size;
            m_ptr = std::exchange(ptr.m_ptr, nullptr);
            is_owner = ptr.is_owner;
            return *this;
        }

        [[nodiscard]] inline constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr Type* data() noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr const Type* data() const noexcept { return m_ptr; }

        [[nodiscard]] inline constexpr size_t size() const noexcept { return m_size; }
        [[nodiscard]] inline constexpr size_t bytes() const noexcept { return m_size * sizeof(Type); }
        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return m_ptr; }

        inline constexpr Type* begin() noexcept { return m_ptr; }
        inline constexpr const Type* begin() const noexcept { return m_ptr; }
        inline constexpr Type* end() noexcept { return m_ptr + m_size; }
        inline constexpr const Type* end() const noexcept { return m_ptr + m_size; }

        inline constexpr std::reverse_iterator<Type> rbegin() noexcept { return m_ptr; }
        inline constexpr std::reverse_iterator<const Type> rbegin() const noexcept { return m_ptr; }
        inline constexpr std::reverse_iterator<Type> rend() noexcept { return m_ptr + m_size; }
        inline constexpr std::reverse_iterator<const Type> rend() const noexcept { return m_ptr + m_size; }

        inline constexpr Type& operator[](size_t idx) noexcept { return *(m_ptr + idx); }
        inline constexpr const Type& operator[](size_t idx) const noexcept { return *(m_ptr + idx); }

        /** Clears the underlying data if necessary. */
        inline void reset() noexcept { dealloc_(); }

        /**
         * Resets the underlying data.
         * @param[in] size      Number of @a Type elements in @a ptr.
         * @param[in] ptr       PtrHost to hold on. If it is not a nullptr, it should correspond to @a size.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        inline void reset(size_t size, Type* ptr, bool own_ptr = false) noexcept {
            dealloc_();
            m_size = size;
            m_ptr = ptr;
            is_owner = own_ptr;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        inline Type* release() noexcept {
            is_owner = false;
            return std::exchange(m_ptr, nullptr);
        }

        /** Returns a human-readable description of the PtrHost. */
        [[nodiscard]] inline std::string toString() const {
            return String::format("Size: {}, Resource: host, Type: {}, Owner: {}, Address: {}",
                                  m_size, String::typeName<Type>(), is_owner, m_ptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~PtrHost() { dealloc_(); }

    private:
        // Allocates. Otherwise, returns nullptr.
        [[nodiscard]] static inline Type* alloc_(size_t size) noexcept {
            return new(std::nothrow) Type[size];
        }

        // Copies the underlying data, preserving the size and the resource.
        [[nodiscard]] inline Type* copy_() noexcept {
            Type* out = new(std::nothrow) Type[size()];
            if (out)
                std::memcpy(out, m_ptr, bytes());
            return out;
        }

        // Deallocates the underlying data, if any and if the instance is the owner.
        inline void dealloc_() noexcept {
            if (is_owner && m_ptr)
                delete[] m_ptr;
        }
    };
}
