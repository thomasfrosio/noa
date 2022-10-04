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
//      data exchange between host and device.
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
    // Manages a page-locked pointer.
    template<typename T>
    class PtrPinned {
    public:
        struct Deleter {
            void operator()(void* ptr) noexcept {
                [[maybe_unused]] const cudaError_t err = cudaFreeHost(ptr);
                NOA_ASSERT(err == cudaSuccess);
            }
        };

    public:
        using alloc_unique_t = unique_t<T[], Deleter>;
        static constexpr size_t ALIGNMENT = 256;

    public: // static functions
        // Allocates pinned memory using cudaMallocHost.
        static alloc_unique_t alloc(dim_t elements) {
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            NOA_THROW_IF(cudaMallocHost(&tmp, elements * sizeof(T)));
            return {static_cast<T*>(tmp), Deleter{}};
        }

    public:
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrPinned() = default;
        constexpr /*implicit*/ PtrPinned(std::nullptr_t) {}

        // Allocates elements of type T on page-locked memory using cudaMallocHost.
        explicit PtrPinned(dim_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

    public: // Getters
        // Returns the host pointer.
        [[nodiscard]] constexpr T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() const noexcept { return m_ptr.get(); }

        // Returns a reference of the shared object.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        // Attach the lifetime of the managed object with alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists. This functions performs no
        // heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename U>
        [[nodiscard]] constexpr std::shared_ptr<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        // How many elements of type are pointed by the managed object.
        [[nodiscard]] constexpr dim_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr dim_t size() const noexcept { return m_elements; }

        // Returns the shape of the allocated data as a row vector.
        [[nodiscard]] constexpr dim4_t shape() const noexcept { return {1, 1, 1, m_elements}; }

        // Returns the strides of the allocated data as a C-contiguous row vector.
        [[nodiscard]] constexpr dim4_t strides() const noexcept { return shape().strides(); }

        // How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        // Whether the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        // Returns a View of the allocated data as a C-contiguous row vector.
        template<typename I>
        [[nodiscard]] constexpr View<T, I> view() const noexcept { return {m_ptr.get(), shape(), strides()}; }

        // Returns a 1D Accessor of the allocated data.
        template<typename U = T, typename I = dim_t, AccessorTraits TRAITS = AccessorTraits::DEFAULT,
                 typename = std::enable_if_t<traits::is_almost_same_v<U, T>>>
        [[nodiscard]] constexpr Accessor<U, 1, I, TRAITS> accessor() const noexcept { return {m_ptr.get(), 1}; }

    public: // Iterators
        [[nodiscard]] constexpr T* begin() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* end() const noexcept { return m_ptr.get() + m_elements; }

        [[nodiscard]] constexpr T& front() const noexcept { return *begin(); }
        [[nodiscard]] constexpr T& back() const noexcept { return *(end() - 1); }

    public: // Accessors
        // Returns a reference at index. There's no bound check.
        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] constexpr T& operator[](I idx) const { return m_ptr.get()[idx]; }

        // Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{nullptr};
        dim_t m_elements{0};
    };
}
