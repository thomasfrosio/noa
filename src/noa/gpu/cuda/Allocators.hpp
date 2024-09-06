#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

#ifdef NOA_IS_OFFLINE
#include <utility> // std::exchange
#include <memory> // std::unique_ptr, std::shared_ptr

#include "noa/core/types/Pair.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Runtime.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda {
    struct AllocatorDeviceDeleter {
        using stream_type = Stream::Core;
        std::weak_ptr<stream_type> stream{};

        void operator()(void* ptr) const noexcept {
            const std::shared_ptr<stream_type> stream_ = stream.lock();
            [[maybe_unused]] cudaError_t err{};
            if (not stream_) {
                // No stream, so the memory was allocated:
                //  - with cudaMalloc, so cudaFree syncs the device
                //  - with cudaMallocAsync, but the stream was deleted, so cudaFree instead
                err = cudaFree(ptr);
            } else {
                err = cudaFreeAsync(ptr, stream_->stream_handle);
            }
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    /// Allocates (global) device memory, using either:
    ///  1) device-wide allocations via cudaMalloc.
    ///  2) stream-ordered allocations via cudaMallocAsync (recommended).
    ///
    /// Stream-ordered allocations: (since CUDA 11.2)
    ///  - https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
    ///  - Memory allocation and deallocation cannot fail asynchronously. Memory errors that occur because of a call to
    ///    cudaMallocAsync or cudaFreeAsync (for example, out of memory) are reported immediately through an error code
    ///    returned from the call. If cudaMallocAsync completes successfully, the returned pointer is guaranteed to be
    ///    a valid pointer to memory that is safe to access in the appropriate stream order.
    ///
    /// Interoperability with cudaMalloc and cudaFree
    ///  - An application can use cudaFreeAsync to free a pointer allocated by cudaMalloc. The underlying memory is not
    ///    freed until the next synchronization of the stream passed to cudaFreeAsync.
    ///  - Similarly, an application can use cudaFree to free memory allocated using cudaMallocAsync. However, cudaFree
    ///    does not implicitly synchronize in this case, so the application must insert the appropriate synchronization
    ///    to ensure that all accesses to the to-be-freed memory are complete.
    template<typename T>
    class AllocatorDevice {
    public:
        static_assert(not std::is_pointer_v<T> and not std::is_reference_v<T> and not std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorDeviceDeleter;
        using shared_type = std::shared_ptr<value_type[]>;
        using unique_type = std::unique_ptr<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // in bytes, this is guaranteed by cuda

    public:
        /// Allocates device memory using cudaMalloc, with an alignment of at least 256 bytes.
        /// This function throws if the allocation fails.
        static unique_type allocate(i64 n_elements, Device device = Device::current()) {
            if (n_elements <= 0)
                return {};
            const DeviceGuard guard(device);
            void* tmp{nullptr}; // X** to void** is not allowed
            check(cudaMalloc(&tmp, static_cast<size_t>(n_elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp));
        }

        /// Allocates device memory asynchronously using cudaMallocAsync, with an alignment of at least 256 bytes.
        /// This function throws if the allocation fails.
        static unique_type allocate_async(i64 n_elements, Stream& stream) {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            check(cudaMallocAsync(&tmp, static_cast<size_t>(n_elements) * sizeof(value_type), stream.id()));
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
        }

        /// Returns the stream handle used to allocate the resource.
        /// If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<nt::any_of<shared_type, unique_type> R>
        [[nodiscard]] static cudaStream_t attached_stream_handle(const R& resource) {
            if (resource) {
                const std::shared_ptr<Stream::Core> stream;
                if constexpr (std::is_same_v<R, shared_type>)
                    stream = std::get_deleter<AllocatorDeviceDeleter>(resource)->stream.lock();
                else
                    resource.get_deleter().stream.lock();
                if (stream)
                    return stream->stream_handle;
            }
            return nullptr;
        }
    };

    struct AllocatorDevicePaddedDeleter {
        void operator()(void* ptr) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFree(ptr); // if nullptr, it does nothing
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    /// Manages a device pointer.
    /// Padded layouts / Pitch memory:
    ///  - AllocatorDevice is allocating "linear" regions, as opposed to AllocatorDevicePadded, which allocates
    ///    "padded" regions. This padding is on the right side of the innermost dimension (i.e. the height,
    ///    in our case). The size of the innermost dimension, including the padding, is called the pitch.
    ///    "Padded" layouts can be useful to minimize the number of memory accesses on a given row (but can increase
    ///    the number of memory accesses for reading the whole array) and to reduce shared memory bank conflicts.
    ///    It is highly recommended to use padded layouts when per-row accesses are necessary, e.g. when the array
    ///    is logically treated as a series of rows, because the alignment at the beginning of every row is preserved.
    ///    See https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
    ///
    /// Pitch:
    ///  - As a result, AllocatorDevicePadded returns the strides (as always, this is in number of elements) of the
    ///    allocated region, taking into account the pitch. Note that this could be an issue since CUDA does not
    ///    guarantee the pitch (originally returned in bytes) to be divisible by the type alignment requirement.
    ///    However, it looks like all devices will return a pitch divisible by at least 16 bytes (which makes sense),
    ///    which is the maximum size allowed by AllocatorDevicePadded. To be safe, AllocatorDevicePadded::allocate()
    ///    checks if this assumption holds.
    template<typename T>
    class AllocatorDevicePadded {
    public:
        static_assert(not std::is_pointer_v<T> and
                      not std::is_reference_v<T> and
                      not std::is_const_v<T> and
                      sizeof(T) <= 16);
        using value_type = T;
        using deleter_type = AllocatorDevicePaddedDeleter;
        using shared_type = std::shared_ptr<value_type[]>;
        using unique_type = std::unique_ptr<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by cuda

    public:
        /// Allocates device memory using cudaMalloc3D.
        /// Returns 1: Unique pointer pointing to the device memory.
        ///         2: Pitch, i.e. height stride, in number of elements.
        template<typename Integer, size_t N> requires (N >= 2)
        static auto allocate(
            const Shape<Integer, N>& shape, // ((B)D)HW order
            Device device = Device::current()
        ) -> Pair<unique_type, Strides<Integer, N>> {
            if (shape.is_empty())
                return {};

            // Get the extents from the shape.
            const auto s_shape = shape.template as_safe<size_t>();
            const cudaExtent extent{s_shape[N - 1] * sizeof(value_type), s_shape.pop_back().n_elements(), 1};

            // Allocate.
            cudaPitchedPtr pitched_ptr{};
            const DeviceGuard guard(device);
            check(cudaMalloc3D(&pitched_ptr, extent));

            if (not is_multiple_of(pitched_ptr.pitch, sizeof(value_type))) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                panic("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                      ns::stringify<value_type>(), pitched_ptr.pitch, sizeof(value_type));
            }

            // Create the strides.
            const auto pitch = static_cast<Integer>(pitched_ptr.pitch / sizeof(value_type));
            Strides<Integer, N> strides = shape.template set<N - 1>(pitch).strides();

            return {unique_type(static_cast<value_type*>(pitched_ptr.ptr)), strides};
        }
    };

    struct AllocatorManagedDeleter {
        std::weak_ptr<Stream::Core> stream{};

        void operator()(void* ptr) const noexcept {
            const std::shared_ptr<Stream::Core> stream_ = stream.lock();
            [[maybe_unused]] cudaError_t err;
            if (stream_) {
                err = cudaStreamSynchronize(stream_->stream_handle);
                NOA_ASSERT(err == cudaSuccess);
            }
            err = cudaFree(ptr);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    /// Unified memory:
    ///  - Managed memory is interoperable and interchangeable with device-specific allocations, such as those created
    ///    using the cudaMalloc() routine. All CUDA operations that are valid on device memory are also valid on managed
    ///    memory; the primary difference is that the host portion of a program is able to reference and access the
    ///    memory as well.
    ///
    ///  - If the stream used by cudaStreamAttachMemAsync is destroyed while data is associated with it, the association is
    ///    removed and the association reverts to the host visibility only. Since destroying a stream is an asynchronous
    ///    operation, the change to default association won't happen until all work in the stream has completed.
    ///
    ///  - Data movement still happens, of course. On compute capabilities >= 6.X, page faulting means that the CUDA
    ///    system software doesn't need to synchronize all managed memory allocations to the GPU before each kernel
    ///    launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing
    ///    the page to be automatically migrated to the GPU memory on-demand. The same thing occurs with CPU page faults.
    ///
    ///  - GPU memory over-subscription: On compute capabilities >= 6.X, applications can allocate and access more
    ///    managed memory than the physical size of GPU memory.
    /// TODO Add prefetching and advising.
    template<typename T>
    class AllocatorManaged {
    public:
        static_assert(not std::is_pointer_v<T> and not std::is_reference_v<T> and not std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorManagedDeleter;
        using shared_type = std::shared_ptr<value_type[]>;
        using unique_type = std::unique_ptr<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by the driver

    public:
        /// Allocates managed memory using cudaMallocManaged, accessible from any stream and any device.
        static unique_type allocate_global(i64 n_elements) {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            check(cudaMallocManaged(&tmp, static_cast<size_t>(n_elements) * sizeof(value_type), cudaMemAttachGlobal));
            return unique_type(static_cast<value_type*>(tmp));
        }

        /// Allocates managed memory using cudaMallocManaged.
        /// The allocation is initially invisible to devices, ensuring that there's no interaction with
        /// thread's execution in the interval between the data allocation and when the data is acquired
        /// by the stream. The program makes a guarantee that it will only access the memory on the device
        /// from stream.
        /// The stream on which to attach the memory should be passed. The returned memory should only be accessed
        /// by the host, and the stream's device from kernels launched with this stream. Note that if the null stream
        /// is passed, the allocation falls back to allocate_global() and the memory can be accessed by any stream
        /// on any device.
        static unique_type allocate(i64 n_elements, Stream& stream) {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (not stream.id())
                return allocate_global(n_elements);
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            check(cudaMallocManaged(&tmp, static_cast<size_t>(n_elements) * sizeof(value_type), cudaMemAttachHost));
            check(cudaStreamAttachMemAsync(stream.id(), tmp));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
        }

        /// Returns the stream handle used to allocate the resource.
        /// If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<nt::any_of< shared_type, unique_type> R>
        [[nodiscard]] cudaStream_t attached_stream_handle(const R& resource) const {
            if (resource) {
                const std::shared_ptr<Stream::Core> stream;
                if constexpr (std::is_same_v<R, shared_type>)
                    stream = std::get_deleter<AllocatorManagedDeleter>(resource)->stream.lock();
                else
                    resource.get_deleter().stream.lock();
                if (stream)
                    return stream->stream_handle;
            }
            return nullptr;
        }

        /// Prefetches the memory region to the stream's GPU.
        static void prefetch_to_gpu(const value_type* pointer, i64 n_elements, Stream& stream) {
            check(cudaMemPrefetchAsync(
                pointer, static_cast<size_t>(n_elements) * sizeof(value_type),
                stream.device().id(), stream.id()));
        }

        /// Prefetches the memory region to the cpu.
        static void prefetch_to_cpu(const value_type* pointer, i64 n_elements, Stream& stream) {
            check(cudaMemPrefetchAsync(
                pointer, static_cast<size_t>(n_elements) * sizeof(value_type),
                cudaCpuDeviceId, stream.id()));
        }
    };

    struct AllocatorPinnedDeleter {
        void operator()(void* ptr) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFreeHost(ptr);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    /// Allocates page-locked memory.
    /// - Page-locked memory is accessible to the device. The driver tracks the virtual memory ranges allocated
    ///   by cudaMallocHost and automatically accelerates calls to functions such as ::cudaMemcpy*(). Since the
    ///   memory can be accessed directly by the device, it can be read or written with much higher bandwidth
    ///   than pageable memory obtained with functions such as ::malloc(). Allocating excessive amounts of memory
    ///   with ::cudaMallocHost() may degrade system performance, since it reduces the amount of memory available
    ///   to the system for paging. As a result, AllocatorPinned is best used sparingly to allocate staging areas for
    ///   data exchange between host and device.
    ///
    /// - For now, the API doesn't include cudaHostRegister, since it is unlikely to be used. However, one can
    ///   still (un)register manually and store the pointer in a AllocatorPinned object if necessary.
    ///
    /// - cudaMallocHost is used, as opposed to cudaHostMalloc since the default flags are enough in most cases.
    ///   See https://stackoverflow.com/questions/35535831.
    ///   -> Portable memory:
    ///      by default, the benefits of using page-locked memory are only available in conjunction
    ///      with the device that was current when the block was allocated (and with all devices sharing
    ///      the same unified address space). The flag `cudaHostAllocPortable` makes it available
    ///      to all devices. Solution: pinned memory is per device, since devices are unlikely to
    ///      work on the same data...
    ///   -> Write-combining memory:
    ///      by default page-locked host memory is allocated as cacheable. It can optionally be
    ///      allocated as write-combining instead by passing flag `cudaHostAllocWriteCombined`.
    ///      It frees up the host's L1 and L2 cache resources, making more cache available to the
    ///      rest of the application. In addition, write-combining memory is not snooped during
    ///      transfers across the PCI Express bus, which can improve transfer performance by up
    ///      to 40%. Reading from write-combining memory from the host is prohibitively slow, so
    ///      write-combining memory should in general be used for memory that the host only
    ///      writes to. Solution: pinned memory is often used as a staging area. Use one area
    ///      for transfer to device and another for transfer from device, so that all transfers
    ///      can be async. Note: In case where there's a lot of devices, we'll probably want to
    ///      restrict the use of pinned memory.
    template<typename T>
    class AllocatorPinned {
    public:
        static_assert(not std::is_pointer_v<T> and not std::is_reference_v<T> and not std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorPinnedDeleter;
        using shared_type = std::shared_ptr<value_type[]>;
        using unique_type = std::unique_ptr<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by cuda

    public:
        // Allocates pinned memory using cudaMallocHost.
        static unique_type allocate(i64 n_elements) {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            check(cudaMallocHost(&tmp, static_cast<size_t>(n_elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp));
        }
    };
}

#endif
