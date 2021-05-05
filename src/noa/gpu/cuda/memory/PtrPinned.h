#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

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
 *      See https://stackoverflow.com/questions/35535831.
 *      - Portable memory: by default, the benefits of using page-locked memory are only available in conjunction
 *                         with the device that was current when the block was allocated (and with all devices sharing
 *                         the same unified address space). The flag `cudaHostAllocPortable` makes it available
 *                         to all devices.
 *                         Solution: pinned memory is per device, since devices won't work on the same data...
 *      - Write-combining memory: by default page-locked host memory is allocated as cacheable. It can optionally be
 *                                allocated as write-combining instead by passing flag `cudaHostAllocWriteCombined`.
 *                                It frees up the host's L1 and L2 cache resources, making more cache available to the
 *                                rest of the application. In addition, write-combining memory is not snooped during
 *                                transfers across the PCI Express bus, which can improve transfer performance by up
 *                                to 40%. Reading from write-combining memory from the host is prohibitively slow, so
 *                                write-combining memory should in general be used for memory that the host only
 *                                writes to.
 *                                Solution: pinned memory is often used as a staging area. Use one area for transfer
 *                                to device and another for transfer from device, so that all transfers can be async.
 *                                Note: In case where there's a lot of devices, we'll probably want to restrict the
 *                                use of pinned memory.
 */

namespace Noa::CUDA::Memory {
    /**
     * Manages a page-locked pointer. This object cannot be used on the device and is not copyable.
     * @tparam Type     Type of the underlying pointer. Anything allowed by @c Traits::is_valid_ptr_type.
     * @throw           @c Noa::Exception, if an error occurs when the data is allocated or freed.
     */
    template<typename Type>
    class PtrPinned {
    private:
        size_t m_elements{0};
        std::enable_if_t<Noa::Traits::is_valid_ptr_type_v<Type>, Type*> m_ptr{nullptr};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrPinned() = default;

        /**
         * Allocates @a elements elements of type @a Type on page-locked memory using @c cudaMallocHost.
         * @param elements  This is attached to the underlying managed pointer and is fixed for the entire
         *                  life of the object. Use elements() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrPinned(size_t elements) : m_elements(elements) { alloc_(); }

        /**
         * Creates an instance from a existing data.
         * @param[in] pinned_ptr    Device pointer to hold on.
         *                          If it is a nullptr, @a elements should be 0.
         *                          If it is not a nullptr, it should correspond to @a elements.
         * @param elements          Number of @a Type elements in @a pinned_ptr
         */
        NOA_HOST PtrPinned(Type* pinned_ptr, size_t elements) noexcept
                : m_elements(elements), m_ptr(pinned_ptr) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrPinned(PtrPinned<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrPinned<Type>& operator=(PtrPinned<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrPinned(const PtrPinned<Type>& to_copy) = delete;
        PtrPinned<Type>& operator=(const PtrPinned<Type>& to_copy) = delete;

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
        [[nodiscard]] NOA_HOST constexpr Type* begin() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* begin() const noexcept { return m_ptr; }

        /** Returns a pointer pointing at the end + 1 of the managed data. */
        [[nodiscard]] NOA_HOST constexpr Type* end() noexcept { return m_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const Type* end() const noexcept { return m_ptr + m_elements; }

        /** Returns a reference at index @a idx. There's no bound check. */
        NOA_HOST constexpr Type& operator[](size_t idx) noexcept { return *(m_ptr + idx); }
        NOA_HOST constexpr const Type& operator[](size_t idx) const noexcept { return *(m_ptr + idx); }

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
         * @param[in] pinned_ptr    Pinned pointer to hold on.
         *                          If it is a nullptr, @a elements should be 0.
         *                          If it is not a nullptr, it should correspond to @a elements.
         * @param elements          Number of @a Type elements in @a pinned_ptr.
         */
        NOA_HOST void reset(Type* pinned_ptr, size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = pinned_ptr;
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
        NOA_HOST ~PtrPinned() { dealloc_(); }

    private:
        // Allocates device memory. m_elements should be set.
        NOA_HOST void alloc_() {
            void* tmp{nullptr}; // Type** to void** not allowed [-fpermissive]
            NOA_THROW_IF(cudaMallocHost(&tmp, bytes()));
            m_ptr = static_cast<Type*>(tmp);
        }

        // Deallocates the underlying data, if any.
        NOA_HOST void dealloc_() {
            if (!m_ptr)
                return;
            NOA_THROW_IF(cudaFreeHost(m_ptr));
        }
    };
}
