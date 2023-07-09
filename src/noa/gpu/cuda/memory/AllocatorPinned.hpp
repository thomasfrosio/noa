#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"

// - Page-locked memory is accessible to the device. The driver tracks the virtual memory ranges allocated
//   by cudaMallocHost and automatically accelerates calls to functions such as ::cudaMemcpy*(). Since the
//   memory can be accessed directly by the device, it can be read or written with much higher bandwidth
//   than pageable memory obtained with functions such as ::malloc(). Allocating excessive amounts of memory
//   with ::cudaMallocHost() may degrade system performance, since it reduces the amount of memory available
//   to the system for paging. As a result, AllocatorPinned is best used sparingly to allocate staging areas for
//   data exchange between host and device.
//
// - For now, the API doesn't include cudaHostRegister, since it is unlikely to be used. However, one can
//   still (un)register manually and store the pointer in a AllocatorPinned object if necessary.
//
// - cudaMallocHost is used, as opposed to cudaHostMalloc since the default flags are enough in most cases.
//   See https://stackoverflow.com/questions/35535831.
//   -> Portable memory:
//      by default, the benefits of using page-locked memory are only available in conjunction
//      with the device that was current when the block was allocated (and with all devices sharing
//      the same unified address space). The flag `cudaHostAllocPortable` makes it available
//      to all devices. Solution: pinned memory is per device, since devices are unlikely to
//      work on the same data...
//   -> Write-combining memory:
//      by default page-locked host memory is allocated as cacheable. It can optionally be
//      allocated as write-combining instead by passing flag `cudaHostAllocWriteCombined`.
//      It frees up the host's L1 and L2 cache resources, making more cache available to the
//      rest of the application. In addition, write-combining memory is not snooped during
//      transfers across the PCI Express bus, which can improve transfer performance by up
//      to 40%. Reading from write-combining memory from the host is prohibitively slow, so
//      write-combining memory should in general be used for memory that the host only
//      writes to. Solution: pinned memory is often used as a staging area. Use one area
//      for transfer to device and another for transfer from device, so that all transfers
//      can be async. Note: In case where there's a lot of devices, we'll probably want to
//      restrict the use of pinned memory.

namespace noa::cuda::memory {
    struct AllocatorPinnedDeleter {
        void operator()(void* ptr) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFreeHost(ptr);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // Allocates page-locked memory.
    template<typename T>
    class AllocatorPinned {
    public:
        static_assert(!std::is_pointer_v<T> && !std::is_reference_v<T> && !std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorPinnedDeleter;
        using shared_type = Shared<value_type[]>;
        using unique_type = Unique<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by cuda

    public: // static functions
        // Allocates pinned memory using cudaMallocHost.
        static unique_type allocate(i64 elements) {
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            NOA_THROW_IF(cudaMallocHost(&tmp, static_cast<size_t>(elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp));
        }
    };
}
