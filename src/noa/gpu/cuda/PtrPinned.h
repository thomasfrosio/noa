#pragma once

#include <type_traits>
#include <string>

#include "noa/gpu/Base.h"
#include "noa/util/string/Format.h"

/*
 * Notes on the page-locked memory and the implementation in PtrPinned.
 * ====================================================================
 *
 * 1)   Page-locked memory is accessible to the device. The driver tracks the virtual memory ranges allocated
 *      by cudaMallocHost and automatically accelerates calls to functions such as ::cudaMemcpy*(). Since the
 *      memory can be accessed directly by the device, it can be read or written with much higher bandwidth
 *      than pageable memory obtained with functions such as ::malloc(). Allocating excessive amounts of memory
 *      with ::cudaMallocHost() may degrade system performance, since it reduces the amount of memory available
 *      to the system for paging. As a result, PtrPinned is best used sparingly to allocate staging areas for
 *      data exchange between host and device
 *
 * 2)   For now, the API doesn't include cudaHostRegister, since it is unlikely to be used. However, one can
 *      still (un)register manually and create a non-owning PtrPinned object if necessary.
 *
 * 3)   cudaMallocHost is used, as opposed to cudaHostMalloc since the default flags are enough in most cases.
 *      See https://stackoverflow.com/questions/35535831
 */

namespace Noa::CUDA {
    /**
     * Manages a page-locked pointer. This object cannot be used on the device. See PtrDevice for more information.
     * @tparam Type     Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw           @c Noa::Exception, if an error occurs when the device data is allocated or freed.
     */
    template<typename Type>
    class PtrPinned {
    private:
        size_t m_elements{};
        std::enable_if_t<Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_pinned_ptr{nullptr};
    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrPinned() = default;

        /**
         * Allocates @a elements elements of type @a Type on page-locked memory using @c cudaMallocHost.
         * @param elements  This is attached to the underlying managed pointer and is fixed for the entire
         *                  life of the object. Use elements() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrPinned(size_t elements) : m_elements(elements) { alloc_(elements); }

        /**
         * Creates an instance from a existing data.
         * @param elements          Number of @a Type elements in @a pinned_ptr
         * @param[in] pinned_ptr    Device pointer to hold on.
         *                          If it is a nullptr, @a elements should be 0.
         *                          If it is not a nullptr, it should correspond to @a elements.
         * @param owner             Whether or not this new instance should own @a pinned_ptr
         */
        NOA_HOST PtrPinned(size_t elements, Type* pinned_ptr, bool owner) noexcept
                : m_elements(elements), m_pinned_ptr(pinned_ptr), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrPinned(const PtrPinned<Type>& to_copy) noexcept
                : m_elements(to_copy.m_elements), m_pinned_ptr(to_copy.m_pinned_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrPinned(PtrPinned<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_pinned_ptr(std::exchange(to_move.m_pinned_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrPinned<Type>& operator=(const PtrPinned<Type>& to_copy) = delete;
        PtrPinned<Type>& operator=(PtrPinned<Type>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_pinned_ptr; }

        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements() == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return m_pinned_ptr; }

        /** Returns a pointer pointing at the beginning of the managed data. */
        [[nodiscard]] NOA_HOST constexpr Type* begin() noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* begin() const noexcept { return m_pinned_ptr; }

        /** Returns a pointer pointing at the end + 1 of the managed data. */
        [[nodiscard]] NOA_HOST constexpr Type* end() noexcept { return m_pinned_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const Type* end() const noexcept { return m_pinned_ptr + m_elements; }

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
         * @param elements          Number of @a Type elements in @a pinned_ptr.
         * @param[in] pinned_ptr    Pinned pointer to hold on.
         *                          If it is a nullptr, @a elements should be 0.
         *                          If it is not a nullptr, it should correspond to @a elements.
         * @param owner             Whether or not this new instance should own @a pinned_ptr.
         */
        NOA_HOST void reset(size_t elements, Type* pinned_ptr, bool owner) {
            dealloc_();
            m_elements = elements;
            m_pinned_ptr = pinned_ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_elements = 0;
            return std::exchange(m_pinned_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: pinned, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, static_cast<void*>(m_pinned_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrPinned() { dealloc_(); }

    private:
        // Allocates device memory. m_elements should be set.
        NOA_HOST void alloc_() {
            NOA_THROW_IF(cudaMallocHost(&m_pinned_ptr, bytes()));
        }

        // Deallocates the underlying data, if any and if the instance is the owner.
        NOA_HOST void dealloc_() {
            if (!m_pinned_ptr)
                return;
            if (is_owner)
                NOA_THROW_IF(cudaFreeHost(m_pinned_ptr));
            else
                m_pinned_ptr = nullptr;
        }
    };

    template<class T>
    [[nodiscard]] NOA_IH std::string toString(PtrPinned<T> ptr) { return ptr.toString(); }
}

template<typename T>
struct fmt::formatter<Noa::CUDA::PtrPinned<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::CUDA::PtrPinned<T>& ptr, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::CUDA::toString(ptr), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::CUDA::PtrPinned<T>& stream) {
    os << Noa::CUDA::toString(stream);
    return os;
}
