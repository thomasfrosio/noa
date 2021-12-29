#include "noa/common/Types.h"

namespace noa {
    /// Backend of an Array.
    /// \details In this unified interface, the CPU backend is always implied. As such, Backend::CPU implies that only
    ///          the CPU backend should be used, with no GPU support. Backend::CUDA activates GPU support via the CUDA
    ///          backend on top of the CPU backend. Backend::VULKAN and Backend::METAL are currently not supported.
    enum class Backend {
        CPU,
        CUDA,
        VULKAN,
        METAL
    };

    /// Memory resource to use for allocation.
    enum class Resource {
        /// System memory (RAM).
        HOST,

        /// Page-locked system memory (pinned memory).
        /// In noa, this is handled by the current active GPU backend.
        PINNED,

        /// Global device memory.
        DEVICE,
    };

    enum AllocatorFlags : uint32_t {
        HOST = 0,

        /// Page-locked system memory (pinned memory).
        /// In noa, this is handled by the current active GPU backend.
        PINNED = 0b0000'0000,
        PINNED_PORTABLE = 0b0000'0000,
        PINNED_WRITE,

        /// Global device memory.
        DEVICE,
        DEVICE_PADDED
    };

    /// Base allocator for the Array type.
    template<typename T>
    class Allocator {
        /// Allocates memory from a given \p resource.
        /// \param shape    Physical shape (i.e. offsets) to allocate.
        /// \param resource Memory resource to allocate from.
        /// \param at_least Whether or not padding is allowed.
        /// \return         0: Pointer pointing at the start of the allocated region.
        ///                 1: If \p at_least is true, returns the shape that was actually allocated.
        ///                    Otherwise, returns \p shape.
        /// \note If \p at_least is true, the returned shape is guaranteed to be equal of larger than \p shape.
        static std::pair<T*, size4_t> allocate(size4_t shape, Resource resource, bool at_least);

        /// Deallocates the storage referenced by the pointer zp data, which must be a pointer obtained by
        /// an earlier call to allocate().
        /// \param[in,out] data Pointer obtained from allocate().
        /// \param shape        Physical shape (i.e. offsets) earlier passed or returned by allocate().
        /// \param resource     Memory resource used by allocate().
        static void deallocate(T* data, size4_t shape, Resource resource);
    };
}
