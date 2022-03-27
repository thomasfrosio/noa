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
    /// Manages a page-locked pointer.
    template<typename T>
    class PtrPinned {
    public:
        struct Deleter {
            void operator()(void* ptr) noexcept {
                cudaFreeHost(ptr);
            }
        };

    public: // static functions
        /// Allocates pinned memory using cudaMallocHost.
        /// \param elements     Number of elements to allocate.
        /// \return             Pointer pointing to pinned memory.
        static std::unique_ptr<T[], Deleter> alloc(size_t elements) {
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            NOA_THROW_IF(cudaMallocHost(&tmp, elements * sizeof(T)));
            return {static_cast<T*>(tmp), Deleter{}};
        }

    public:
        /// Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrPinned() = default;
        constexpr /*implicit*/ PtrPinned(std::nullptr_t) {}

        /// Allocates \p elements elements of type \p T on page-locked memory using \c cudaMallocHost.
        explicit PtrPinned(size_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

    public:
        /// Returns the pinned pointer.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr std::shared_ptr<T[]>& share() noexcept { return m_ptr; }
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        [[nodiscard]] constexpr T* begin() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* begin() const noexcept { return m_ptr.get(); }

        /// Returns a pointer pointing at the end + 1 of the managed data.
        [[nodiscard]] constexpr T* end() noexcept { return m_ptr.get() + m_elements; }
        [[nodiscard]] constexpr const T* end() const noexcept { return m_ptr.get() + m_elements; }

        /// Returns a reference at index \p idx. There's no bound check.
        constexpr T& operator[](size_t idx) noexcept { return m_ptr[idx]; }
        constexpr const T& operator[](size_t idx) const noexcept { return m_ptr[idx]; }

        /// Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(noa::traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{nullptr};
        size_t m_elements{0};
    };
}
