/// \file noa/gpu/cuda/memory/PtrPinned.h
/// \brief Hold paged-locked memory on the host.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// Notes on the page-locked memory and the implementation in PtrPinned.
// ====================================================================
//
// 1)   Page-locked memory is accessible to the device. The driver tracks the virtual memory ranges allocated
//      by cudaMallocHost and automatically accelerates calls to functions such as ::cudaMemcpy*(). Since the
//      memory can be accessed directly by the device, it can be read or written with much higher bandwidth
//      than pageable memory obtained with functions such as ::malloc(). Allocating excessive amounts of memory
//      with ::cudaMallocHost() may degrade system performance, since it reduces the amount of memory available
//      to the system for paging. As a result, PtrPinned is best used sparingly to allocate staging areas for
//      data exchange between host and device
//
// 2)   For now, the API doesn't include cudaHostRegister, since it is unlikely to be used. However, one can
//      still (un)register manually and store the pointer in a PtrPinned object if necessary.
//
// 3)   cudaMallocHost is used, as opposed to cudaHostMalloc since the default flags are enough in most cases.
//      See https://stackoverflow.com/questions/35535831.
//      - Portable memory: by default, the benefits of using page-locked memory are only available in conjunction
//                         with the device that was current when the block was allocated (and with all devices sharing
//                         the same unified address space). The flag `cudaHostAllocPortable` makes it available
//                         to all devices.
//                         Solution: pinned memory is per device, since devices are unlikely to work on the same data...
//      - Write-combining memory: by default page-locked host memory is allocated as cacheable. It can optionally be
//                                allocated as write-combining instead by passing flag `cudaHostAllocWriteCombined`.
//                                It frees up the host's L1 and L2 cache resources, making more cache available to the
//                                rest of the application. In addition, write-combining memory is not snooped during
//                                transfers across the PCI Express bus, which can improve transfer performance by up
//                                to 40%. Reading from write-combining memory from the host is prohibitively slow, so
//                                write-combining memory should in general be used for memory that the host only
//                                writes to.
//                                Solution: pinned memory is often used as a staging area. Use one area for transfer
//                                to device and another for transfer from device, so that all transfers can be async.
//                                Note: In case where there's a lot of devices, we'll probably want to restrict the
//                                use of pinned memory.

namespace noa::cuda::memory {
    /// Manages a page-locked pointer. This object is not copyable.
    /// \tparam T   Type of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type.
    template<typename T>
    class PtrPinned {
    public: // static functions
        /// Allocates pinned memory using cudaMallocHost.
        /// \param elements     Number of elements to allocate.
        /// \return             Pointer pointing to pinned memory.
        NOA_HOST static T* alloc(size_t elements) {
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            NOA_THROW_IF(cudaMallocHost(&tmp, elements * sizeof(T)));
            return static_cast<T*>(tmp);
        }

        /// Deallocates pinned memory allocated by the cudaMallocHost functions.
        /// \param[out] ptr     Pointer pointing to pinned memory, or nullptr.
        NOA_HOST static void dealloc(T* ptr) {
            NOA_THROW_IF(cudaFreeHost(ptr));
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrPinned() = default;

        /// Allocates \p elements elements of type \p T on page-locked memory using \c cudaMallocHost.
        /// \param elements     Number of elements to allocate.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        NOA_HOST explicit PtrPinned(size_t elements) : m_elements(elements) {
            m_ptr = alloc(elements);
        }

        /// Creates an instance from a existing data.
        /// \param[in] data Pointer pointing at pinned memory to hold on.
        /// \param elements Number of \p T elements in \p data
        NOA_HOST PtrPinned(T* data, size_t elements) noexcept
                : m_elements(elements), m_ptr(data) {}

        /// Move constructor. \p to_move is not meant to be used after this call.
        NOA_HOST PtrPinned(PtrPinned<T>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \p to_move is not meant to be used after this call.
        NOA_HOST PtrPinned<T>& operator=(PtrPinned<T>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrPinned(const PtrPinned<T>& to_copy) = delete;
        PtrPinned<T>& operator=(const PtrPinned<T>& to_copy) = delete;

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
        [[nodiscard]] NOA_HOST constexpr T* begin() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* begin() const noexcept { return m_ptr; }

        /// Returns a pointer pointing at the end + 1 of the managed data.
        [[nodiscard]] NOA_HOST constexpr T* end() noexcept { return m_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const T* end() const noexcept { return m_ptr + m_elements; }

        /// Returns a reference at index \p idx. There's no bound check.
        NOA_HOST constexpr T& operator[](size_t idx) noexcept { return *(m_ptr + idx); }
        NOA_HOST constexpr const T& operator[](size_t idx) const noexcept { return *(m_ptr + idx); }

        /// Clears the underlying data, if necessary. empty() will evaluate to true.
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
        /// \param[in] data Pinned pointer to hold on.
        /// \param elements Number of \p T elements in \p data.
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
        NOA_HOST ~PtrPinned() noexcept(false) {
            cudaError_t err = cudaFreeHost(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

    private:
        size_t m_elements{0};
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<T>, T*> m_ptr{nullptr};
    };
}
