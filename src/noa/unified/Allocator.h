#pragma once

#include <ostream>
#include "noa/common/Definitions.h"

namespace noa {
    /// Memory allocators.
    /// Memory allocation depends on the device used for the allocation.
    enum class Allocator {
        /// No allocation is performed.
        NONE = 0,

        /// The device default allocator.
        /// - \b Allocation: For CPUs, it refers to the standard allocator using the heap as resource and
        ///   returning at least 64-bytes aligned pointer. For GPUs, it refers to the GPU backend default
        ///   allocator using the GPU global memory as resource. In CUDA, pointers have a minimum 256-bytes
        ///   alignment. Allocations do not use the current stream.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device.
        DEFAULT = 1,

        /// The device asynchronous allocator.
        /// - \b Allocation: Same as DEFAULT, except if the device is a CUDA-capable device. In this case,
        ///   the current stream of the device is used to performed the allocation, which is thereby stream-
        ///   ordered. Since CUDA 11.2, it is the recommend way to allocate GPU memory. The alignment is
        ///   the same as DEFAULT.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device. If the device is a CUDA-capable device, one
        ///   should make sure the memory is accessed in the appropriate stream-order after allocation since
        ///   the memory is only valid when the stream reaches the allocation event.
        DEFAULT_ASYNC = 2,

        /// "Pitch" allocator.
        /// - \b Allocation: This is equivalent to DEFAULT, except for CUDA-capable devices. In this case,
        ///   the CUDA driver will potentially pad the right side of the innermost dimension of the ND array.
        ///   The size of the innermost dimension, including the padding, is called the "pitch". "Pitched"
        ///   layouts can be useful to minimize the number of memory accesses on a given row (but can increase
        ///   the number of memory accesses for reading the whole array) and to reduce shared memory bank
        ///   conflicts. It is highly recommended to use these layouts if the application will be performing
        ///   memory copies involving 2D or 3D CUDA arrays. Allocations do not use the current stream.
        /// - \b Accessibility: Allocated memory is private to the device that performed the allocation,
        ///   but can be used by any stream of that device.
        PITCHED = 4,

        /// Page-locked (i.e. pinned) memory allocator.
        /// - \b Allocation: Pinned memory can be allocated by a CPU or a GPU device. Allocating excessive
        ///   amounts of pinned memory may degrade system performance, since it reduces the amount of memory
        ///   available to the system for paging. Thus, it is best used sparingly, e.g. to allocate staging
        ///   areas for data exchange between CPU and GPU. Allocations do not use the current stream.
        /// - \b Accessibility: Can be accessed by the CPU, and the GPU against which the allocation was
        ///   performed. If the CPU device was used for allocation, this GPU is the "current" GPU at the
        ///   time of allocation.
        PINNED = 8,

        /// Managed memory allocator.
        /// - \b Allocation: If the device is the CPU, the current GPU stream of the current GPU is used to
        ///   perform the allocation. Otherwise, the current GPU stream of the GPU device is used. While
        ///   streams are used (the memory is attached to them), the allocation itself is synchronous.
        /// - \b Accessibility: Can be accessed by the CPU. If the GPU stream used for the allocation
        ///   was the NULL stream, this is equivalent to MANAGED_GLOBAL. Otherwise, the allocated memory on
        ///   the GPU side is private to the stream and the GPU that performed the allocation.
        MANAGED = 16,

        /// Managed memory allocator.
        /// - \b Allocation: Managed memory can be allocated by a CPU or a GPU device. Allocation does not
        ///   use the current stream. Note that this is much less efficient compared to a stream-private
        ///   allocation with MANAGED.
        /// - \b Accessibility: Can be accessed by any stream and any device (CPU and GPU).
        MANAGED_GLOBAL = 32,

        /// CUDA array.
        /// - \b Allocation: This is only supported by CUDA-capable devices and is only used for textures.
        /// - \b Accessibility: Can only be accessed via texture fetching on the device it was allocated on.
        CUDA_ARRAY = 64
    };

    inline std::ostream& operator<<(std::ostream& os, Allocator resource) {
        switch (resource) {
            case Allocator::NONE:
                return os << "NONE";
            case Allocator::DEFAULT:
                return os << "DEFAULT";
            case Allocator::DEFAULT_ASYNC:
                return os << "DEFAULT_ASYNC";
            case Allocator::PITCHED:
                return os << "PITCHED";
            case Allocator::PINNED:
                return os << "PINNED";
            case Allocator::MANAGED:
                return os << "MANAGED";
            case Allocator::MANAGED_GLOBAL:
                return os << "MANAGED_GLOBAL";
            case Allocator::CUDA_ARRAY:
                return os << "CUDA_ARRAY";
        }
        return os;
    }
}
