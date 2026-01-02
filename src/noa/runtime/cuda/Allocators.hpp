#pragma once

#include <memory> // std::unique_ptr, std::shared_ptr

#include "noa/base/Complex.hpp"
#include "noa/base/Half.hpp"
#include "noa/base/Pair.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cuda/Device.hpp"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Runtime.hpp"
#include "noa/runtime/cuda/Stream.hpp"

// Add specialization for our complex types. Used for CUDA arrays and textures.
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

namespace noa::cuda {
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
    class AllocatorDevice {
    public:
        static constexpr usize MIN_ALIGNMENT = 256; // this is guaranteed by cuda

        struct Deleter {
            std::weak_ptr<Stream::Core> stream{};
            isize size{};
            i32 device_id{};

            void operator()(void* ptr) const noexcept {
                if (const auto stream_ = stream.lock()) {
                    [[maybe_unused]] auto err = cudaFreeAsync(ptr, stream_->stream_handle);
                    NOA_ASSERT(err == cudaSuccess);
                } else {
                    // No stream, so the memory was allocated:
                    //  - with cudaMalloc, so cudaFree syncs the device
                    //  - with cudaMallocAsync, but the stream was deleted, so cudaFree instead
                    [[maybe_unused]] auto err = cudaFree(ptr);
                    NOA_ASSERT(err == cudaSuccess);
                }
                add_bytes(device_id, -size);
            }
        };

        template<typename T>
        using allocate_type = std::unique_ptr<T[], Deleter>;

    public:
        template<nt::allocatable_type T>
        static auto try_allocate(isize n_elements, Device device = Device::current()) -> allocate_type<T> {
            if (n_elements <= 0)
                return {};
            const auto guard = DeviceGuard(device);
            void* tmp{nullptr}; // X** to void** is not allowed
            const auto n_bytes = n_elements * static_cast<isize>(sizeof(T));
            if (cudaMalloc(&tmp, static_cast<usize>(n_bytes)) != cudaSuccess)
                return {};
            add_bytes(device.id(), n_bytes);
            return {static_cast<T*>(tmp), Deleter{.size=n_bytes, .device_id=device.id()}};
        }

        /// Allocates device memory using cudaMalloc, with an alignment of at least 256 bytes.
        /// This function throws if the allocation fails.
        template<nt::allocatable_type T>
        static auto allocate(isize n_elements, Device device = Device::current()) -> allocate_type<T> {
            auto ptr = try_allocate<T>(n_elements, device);
            check(ptr);
            return ptr;
        }

        /// Allocates device memory asynchronously using cudaMallocAsync, with an alignment of at least 256 bytes.
        /// This function throws if the allocation fails.
        template<nt::allocatable_type T>
        static auto allocate_async(isize n_elements, Stream& stream) -> allocate_type<T> {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            const auto n_bytes = n_elements * static_cast<isize>(sizeof(T));
            check(cudaMallocAsync(&tmp, static_cast<usize>(n_bytes), stream.id()));
            add_bytes(stream.device().id(), n_bytes);
            return {static_cast<T*>(tmp), Deleter{.stream = stream.core(), .size=n_bytes, .device_id=stream.device().id()}};
        }

        /// Returns the stream handle used to allocate the resource.
        /// If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<nt::allocatable_type T>
        [[nodiscard]] static auto attached_stream_handle(const allocate_type<T>& resource) -> cudaStream_t {
            if (resource) {
                if (auto stream = resource.get_deleter().stream.lock())
                    return stream->stream_handle;
            }
            return nullptr;
        }

        /// Returns the number of bytes currently allocated on the device.
        [[nodiscard]] static auto bytes_currently_allocated(i32 device_id) -> usize {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                return clamp_cast<usize>(m_bytes_currently_allocated[device_id].load());
            return 0;
        }

    private:
        static constexpr i32 MAX_DEVICES = 64;
        static void add_bytes(i32 device_id, isize n_bytes) noexcept {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                m_bytes_currently_allocated[device_id] += n_bytes;
        }
        inline static std::atomic<isize> m_bytes_currently_allocated[MAX_DEVICES]{};
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
    ///
    class AllocatorDevicePadded {
    public:
        static constexpr usize MIN_ALIGNMENT = 256; // this is guaranteed by cuda

        struct Deleter {
            isize size{};
            i32 device_id{};
            void operator()(void* ptr) const noexcept {
                [[maybe_unused]] const auto err = cudaFree(ptr); // if nullptr, it does nothing
                NOA_ASSERT(err == cudaSuccess);
                add_bytes(device_id, -size);
            }
        };

        template<typename T>
        using allocate_type = std::unique_ptr<T[], Deleter>;

    public:
        /// Allocates device memory using cudaMalloc3D.
        /// Returns 1: Unique pointer pointing to the device memory.
        ///         2: Strides describing the allocated layout.
        template<nt::allocatable_type T, typename I, usize N>
        requires (sizeof(T) <= 16 and N >= 2)
        static auto allocate(
            const Shape<I, N>& shape, // ((B)D)HW order
            Device device = Device::current()
        ) -> Pair<allocate_type<T>, Strides<I, N>> {
            if (shape.is_empty())
                return {};

            // Get the extents from the shape.
            const auto s_shape = shape.template as_safe<usize>();
            const auto extent = cudaExtent{s_shape[N - 1] * sizeof(T), s_shape.pop_back().n_elements(), 1};

            // Allocate.
            auto pitched_ptr = cudaPitchedPtr{};
            const auto guard = DeviceGuard(device);
            check(cudaMalloc3D(&pitched_ptr, extent));

            if (not is_multiple_of(pitched_ptr.pitch, sizeof(T))) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                panic("The returned pitch is not divisible by sizeof({}): {} % {} != 0. Please report this issue",
                      nd::stringify<T>(), pitched_ptr.pitch, sizeof(T));
            }

            // Create the strides.
            const auto pitch = static_cast<I>(pitched_ptr.pitch / sizeof(T));
            const auto physical_shape = shape.template set<N - 1>(pitch);
            const auto strides = physical_shape.strides();
            const auto n_bytes = physical_shape.n_elements() * static_cast<isize>(sizeof(T));
            add_bytes(device.id(), n_bytes);

            return {allocate_type<T>(static_cast<T*>(pitched_ptr.ptr), Deleter{.size = n_bytes, .device_id = device.id()}), strides};
        }

        /// Returns the number of bytes currently allocated on the device.
        [[nodiscard]] static auto bytes_currently_allocated(i32 device_id) -> usize {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                return clamp_cast<usize>(m_bytes_currently_allocated[device_id].load());
            return 0;
        }

    private:
        static constexpr i32 MAX_DEVICES = 64;
        static void add_bytes(i32 device_id, isize n_bytes) noexcept {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                m_bytes_currently_allocated[device_id] += n_bytes;
        }
        inline static std::atomic<isize> m_bytes_currently_allocated[MAX_DEVICES]{};
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
    class AllocatorManaged {
    public:
        struct Deleter {
            std::weak_ptr<Stream::Core> stream{};
            isize size{};
            i32 device_id{};

            void operator()(void* ptr) const noexcept {
                if (const auto stream_ = stream.lock()) {
                    [[maybe_unused]] auto err = cudaStreamSynchronize(stream_->stream_handle);
                    NOA_ASSERT(err == cudaSuccess);
                }
                [[maybe_unused]] auto err = cudaFree(ptr);
                NOA_ASSERT(err == cudaSuccess);
                add_bytes(device_id, -size);
            }
        };

        template<typename T>
        using allocate_type = std::unique_ptr<T[], Deleter>;

        static constexpr usize MIN_ALIGNMENT = 256; // this is guaranteed by cuda

    public:
        /// Allocates managed memory using cudaMallocManaged, accessible from any stream and any device.
        template<nt::allocatable_type T>
        static auto allocate_global(isize n_elements) -> allocate_type<T> {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            const auto n_bytes = static_cast<usize>(n_elements) * sizeof(T);
            check(cudaMallocManaged(&tmp, n_bytes, cudaMemAttachGlobal));
            add_bytes(-1, static_cast<isize>(n_bytes));
            return allocate_type<T>(static_cast<T*>(tmp), Deleter{
                .size = static_cast<isize>(n_bytes),
                .device_id = -1,
            });
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
        template<nt::allocatable_type T>
        static auto allocate(isize n_elements, Stream& stream) -> allocate_type<T> {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (not stream.id())
                return allocate_global<T>(n_elements);
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            const auto n_bytes = static_cast<usize>(n_elements) * sizeof(T);
            check(cudaMallocManaged(&tmp, n_bytes, cudaMemAttachHost));
            check(cudaStreamAttachMemAsync(stream.id(), tmp));
            add_bytes(stream.device().id(), static_cast<isize>(n_bytes));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return allocate_type<T>(static_cast<T*>(tmp), Deleter{
                .stream = stream.core(),
                .size = static_cast<isize>(n_bytes),
                .device_id = stream.device().id(),
            });
        }

        /// Returns the stream handle used to allocate the resource.
        /// If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<nt::allocatable_type T>
        [[nodiscard]] auto attached_stream_handle(const allocate_type<T>& resource) const -> cudaStream_t {
            if (resource) {
                if (auto stream = resource.get_deleter().stream.lock())
                    return stream->stream_handle;
            }
            return nullptr;
        }

        /// Prefetches the memory region to the stream's GPU.
        template<nt::allocatable_type T>
        static void prefetch_to_gpu(const T* pointer, isize n_elements, Stream& stream) {
            const auto n_bytes = static_cast<usize>(n_elements) * sizeof(T);
            #if CUDART_VERSION < 13000
            check(cudaMemPrefetchAsync(pointer, n_bytes, stream.device().id(), stream.id()));
            #else
            auto location = cudaMemLocation{.type = cudaMemLocationTypeDevice, .id = stream.device().id()};
            cudaMemPrefetchAsync(pointer, n_bytes, location, {}, stream.id());
            #endif
        }

        /// Prefetches the memory region to the cpu.
        template<nt::allocatable_type T>
        static void prefetch_to_cpu(const T* pointer, isize n_elements, Stream& stream) {
            const auto n_bytes = static_cast<usize>(n_elements) * sizeof(T);
            #if CUDART_VERSION < 13000
            check(cudaMemPrefetchAsync(pointer, n_bytes, cudaCpuDeviceId, stream.id()));
            #else
            auto location = cudaMemLocation{.type = cudaMemLocationTypeHost, .id = 0};
            cudaMemPrefetchAsync(pointer, n_bytes, location, {}, stream.id());
            #endif
        }

        /// Returns the number of bytes currently allocated on the device.
        /// Note that memory allocated using allocate_global is added to every device.
        [[nodiscard]] static auto bytes_currently_allocated(i32 device_id) -> usize {
            auto n_bytes = clamp_cast<usize>(m_bytes_currently_allocated_globally.load());
            if (device_id >= 0 and device_id < MAX_DEVICES)
                n_bytes += clamp_cast<usize>(m_bytes_currently_allocated[device_id].load());
            return n_bytes;
        }

    private:
        static constexpr i32 MAX_DEVICES = 64;
        static void add_bytes(i32 device_id, isize n_bytes) noexcept {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                m_bytes_currently_allocated[device_id] += n_bytes;
            else
                m_bytes_currently_allocated_globally += n_bytes;
        }
        inline static std::atomic<isize> m_bytes_currently_allocated[MAX_DEVICES]{};
        inline static std::atomic<isize> m_bytes_currently_allocated_globally{};
    };

    class AllocatorManagedPadded {
    public:
        static constexpr usize ALIGNMENT = 256; // this is guaranteed by cuda

        template<typename T>
        using allocate_type = AllocatorManaged::allocate_type<T>;

    public:
        /// Allocates managed memory using AllocatorManaged::allocate<T>().
        /// Returns 1: Unique pointer pointing to the managed memory.
        ///         2: Strides describing the allocated layout.
        template<nt::allocatable_type T, typename I, usize N> requires (N >= 2)
        static auto allocate(
            const Shape<I, N>& shape, // ((B)D)HW order
            Stream& stream
        ) -> Pair<allocate_type<T>, Strides<I, N>> {
            if (shape.is_empty())
                return {};

            // Get the pitch from the device.
            auto pitch = static_cast<usize>(stream.device().attribute(cudaDevAttrTexturePitchAlignment));
            pitch = std::max(ALIGNMENT, pitch);
            check(is_multiple_of(pitch, sizeof(T)),
                  "The pitch must be a multiple of sizeof({})={}, but got {}",
                  nd::stringify<T>(), sizeof(T), pitch);

            auto width = next_multiple_of(shape.width(), safe_cast<I>(pitch / sizeof(T)));
            auto padded_shape = shape.template set<N - 1>(width);
            return {AllocatorManaged::allocate<T>(padded_shape.n_elements(), stream), padded_shape.strides()};
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
    ///      By default, the benefits of using page-locked memory are only available in conjunction
    ///      with the device that was current when the block was allocated (and with all devices sharing
    ///      the same unified address space). The flag `cudaHostAllocPortable` makes it available
    ///      to all devices. Solution: pinned memory is per device, since devices are unlikely to
    ///      work on the same data...
    ///   -> Write-combining memory:
    ///      By default, page-locked host memory is allocated as cacheable. It can optionally be
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
    class AllocatorPinned {
    public:
        static constexpr usize ALIGNMENT = 256; // this is guaranteed by cuda

        struct Deleter {
            isize size{};
            i32 device_id{};
            void operator()(void* ptr) const noexcept {
                [[maybe_unused]] const cudaError_t err = cudaFreeHost(ptr);
                NOA_ASSERT(err == cudaSuccess);
                add_bytes(device_id, -size);
            }
        };

        template<typename T>
        using allocate_type = std::unique_ptr<T[], Deleter>;

    public:
        // Allocates pinned memory using cudaMallocHost.
        template<nt::allocatable_type T>
        static auto allocate(isize n_elements, Device device = Device::current()) -> allocate_type<T> {
            if (n_elements <= 0)
                return {};
            void* tmp{nullptr}; // T** to void** not allowed [-fpermissive]
            auto n_bytes = n_elements * static_cast<isize>(sizeof(T));
            auto device_guard = DeviceGuard(device);
            check(cudaMallocHost(&tmp, static_cast<usize>(n_bytes)));
            add_bytes(device_guard.id(), n_bytes);
            return allocate_type<T>(static_cast<T*>(tmp), Deleter{.size = n_bytes, .device_id = device_guard.id()});
        }

        /// Returns the number of bytes currently allocated on the device.
        /// Note that memory allocated using allocate_global is added to every device.
        [[nodiscard]] static auto bytes_currently_allocated(i32 device_id) -> usize {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                return clamp_cast<usize>(m_bytes_currently_allocated[device_id].load());
            return 0;
        }

    private:
        static constexpr i32 MAX_DEVICES = 64;
        static void add_bytes(i32 device_id, isize n_bytes) noexcept {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                m_bytes_currently_allocated[device_id] += n_bytes;
        }
        inline static std::atomic<isize> m_bytes_currently_allocated[MAX_DEVICES]{};
    };

    /// Creates 1d, 2d, or 3d CUDA array.
    /// CUDA arrays:
    ///  - Data resides in global memory. The host can cudaMemcpy to it, and the device can only access it
    ///    through texture reads or surface reads and writes.
    ///  - They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
    ///    2 components). Elements are associated with a type (components have the same type), that may be signed or
    ///    unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
    ///  - They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0 in the CUDA API, but AllocatorArray
    ///    is following our "shape" convention (i.e. "empty" dimensions are noted as 1). See shape2extent.
    ///  - They are cached-optimized for 2D/3D spatial locality.
    ///  - Surfaces and textures can be bound to same CUDA array.
    ///  - They are mostly used when the content changes rarely.
    ///    Although reusing them with cudaMemcpy is possible and surfaces can write to it.
    class AllocatorArray {
    public:
        struct Deleter {
            isize size{};
            i32 device_id{};
            void operator()(cudaArray_t ptr) const noexcept {
                [[maybe_unused]] auto err = cudaFreeArray(ptr);
                NOA_ASSERT(err == cudaSuccess);
                add_bytes(device_id, -size);
            }
        };

        using allocate_type = std::unique_ptr<cudaArray, Deleter>;

    public:
        /// Allocates a CUDA array.
        template<nt::any_of<i8, i16, i32, u8, u16, u32, f16, f32, c16, c32> T>
        static auto allocate(
            const Shape4& shape,
            Device device = Device::current()
        ) -> allocate_type {
            const auto device_guard = DeviceGuard(device);
            const auto desc = cudaCreateChannelDesc<T>();
            const auto is_layered = shape.ndim() == 2;
            const auto extent = shape2extent(shape, is_layered);
            cudaArray_t ptr;
            check(cudaMalloc3DArray(&ptr, &desc, extent, is_layered ? cudaArrayLayered : cudaArrayDefault));

            auto n_bytes = shape.n_elements() * static_cast<isize>(sizeof(T));
            add_bytes(device.id(), n_bytes); // TODO include pitch
            return allocate_type(ptr, Deleter{.size = n_bytes, .device_id = device_guard.id()});
        }

    public: // static array utilities
        static auto shape2extent(Shape4 shape, bool is_layered) -> cudaExtent {
            // Special case: treat column vectors as row vectors.
            if (shape[2] >= 1 and shape[3] == 1)
                std::swap(shape[2], shape[3]);

            // Conversion:  shape -> CUDA extent
            // 3D:          1DHW  -> DHW
            // 2D:          11HW  -> 0HW
            // 1D:          111W  -> 00W
            // 2D layered:  B1HW  -> DHW
            // 1D layered:  B11W  -> D0W
            check(shape > 0 and shape[is_layered] == 1,
                  "The input shape cannot be converted to a CUDA array extent. "
                  "Dimensions with a size of 0 are not allowed, and the {} should be 1. Shape: {}",
                  is_layered ? "depth dimension (for layered arrays)" : "batch dimension", shape);

            auto shape_3d = shape.filter(static_cast<isize>(not is_layered), 2, 3).as_safe<usize>();

            // Set empty dimensions to 0. If layered, leave extent.depth to the batch value.
            if (not is_layered)
                shape_3d[0] -= shape_3d[0] == 1;
            shape_3d[1] -= shape_3d[1] == 1;
            return {shape_3d[2], shape_3d[1], shape_3d[0]};
        }

        static auto extent2shape(cudaExtent extent, bool is_layered) noexcept -> Shape4 {
            auto u_extent = Shape{extent.depth, extent.height, extent.width};
            u_extent += Shape<usize, 3>::from_vec(u_extent.cmp_eq(0)); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1d extents are interpreted as row vectors.
            auto shape = Shape4::from_values(1, 1, u_extent[1], u_extent[2]);
            shape[not is_layered] = static_cast<isize>(u_extent[0]);
            return shape;
        }

        static auto array_info(cudaArray* array) {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            u32 flags{};
            check(cudaArrayGetInfo(&desc, &extent, &flags, array));
            return make_tuple(desc, extent, flags);
        }

        static bool is_layered(cudaArray* array) {
            const auto [desc_, extent_, flags] = array_info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

        /// Returns the number of bytes currently allocated on the device.
        [[nodiscard]] static auto bytes_currently_allocated(i32 device_id) -> usize {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                return clamp_cast<usize>(m_bytes_currently_allocated[device_id].load());
            return 0;
        }

    private:
        static constexpr i32 MAX_DEVICES = 64;
        static void add_bytes(i32 device_id, isize n_bytes) noexcept {
            if (device_id >= 0 and device_id < MAX_DEVICES)
                m_bytes_currently_allocated[device_id] += n_bytes;
        }
        inline static std::atomic<isize> m_bytes_currently_allocated[MAX_DEVICES]{};
    };
}
