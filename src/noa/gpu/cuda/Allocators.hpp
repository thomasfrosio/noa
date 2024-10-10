#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

#ifdef NOA_IS_OFFLINE
#include <utility> // std::exchange
#include <memory> // std::unique_ptr, std::shared_ptr

#include "noa/core/Enums.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Runtime.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// Add specialization for our complex types. Used for CUDA arrays and textures.
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

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

    /// Creates 1d, 2d or 3d texture objects bounded to a CUDA array.
    ///
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
    ///
    /// CUDA textures:
    ///  -   Address mode: How out of range coordinates are handled. This can be specified for each coordinates (although
    ///                    the current implementation specifies the same mode for all the dimensions. It is either wrap,
    ///                    mirror, border or clamp (default).
    ///                    Note: This is ignored for 1D textures since they don't support addressing modes.
    ///                    Note: mirror and wrap are only supported for normalized coordinates, otherwise, fallback to clamp.
    ///  -   Filter mode:  Filtering used when fetching. Either point (neighbour) or linear.
    ///                    Note: The linear mode is only allowed for float types.
    ///                    Note: This is ignored for 1D textures since they don't perform any interpolation.
    ///  -   Read mode:    Whether or not integer data should be converted to floating point when fetching. If signed,
    ///                    returns float within [-1., 1.]. If unsigned, returns float within [0., 1.].
    ///                    Note: This only applies to 8-bit and 16-bit integer formats. 32-bits are not promoted.
    ///  -   Normalized coordinates: Whether or not the coordinates are normalized when fetching.
    ///                              If false (default): textures are fetched using floating point coordinates in range
    ///                                                  [0, N-1], where N is the size of that particular dimension.
    ///                              If true:            textures are fetched using floating point coordinates in range
    ///                                                  [0., 1. -1/N], where N is the size of that particular dimension.
    ///
    /// Textures are bound to global memory, either through a device pointer or a CUDA array.
    /// -- Data in the bounded CUDA array can be updated but texture cache is unchanged until a new kernel is launched.
    /// -- The device pointer or a CUDA array should not be freed while the texture is being used.
    class AllocatorTexture {
    public:
        // Textures can map pitch memory (2d only) and linear memory (1d only), but we don't support these
        // use cases as they are either less performant or are very limited compared to a CUDA array.
        struct Resource {
            cudaArray_t array; // pointer
            cudaTextureObject_t texture; // size_t

            Resource() = default;
            ~Resource() {
                [[maybe_unused]] cudaError_t err;
                err = cudaDestroyTextureObject(texture);
                NOA_ASSERT(err == cudaSuccess);
                err = cudaFreeArray(array);
                NOA_ASSERT(err == cudaSuccess);
            }
        };
        using shared_type = std::shared_ptr<Resource>;

    public:
        /// Allocates a CUDA array and create a texture from that array.
        /// The returned array and texture are configured to work with the interpolation functions
        /// (see convert_to_description() for more details).
        template<nt::any_of<i8, i16, i32, u8, u16, u32, f16, f32, c16, c32> T>
        static auto allocate(
            const Shape4<i64>& shape,
            Interp interp_mode,
            Border border_mode,
            u32 flag = cudaArrayDefault
        ) -> std::shared_ptr<Resource> {
            auto resource = std::make_shared<Resource>();

            // Create the array.
            const cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
            const cudaExtent extent = shape2extent(shape, flag & cudaArrayLayered);
            check(cudaMalloc3DArray(&resource->array, &desc, extent, flag));

            // Create the texture.
            const auto [filter, address, read_mode, normalized_coords] = convert_to_description(interp_mode, border_mode);
            resource->texture = create_texture(resource->array, filter, address, read_mode, normalized_coords);

            return resource;
        }

    public: // static array utilities
        static cudaExtent shape2extent(Shape4<i64> shape, bool is_layered) {
            // Special case: treat column vectors as row vectors.
            if (shape[2] >= 1 and shape[3] == 1)
                std::swap(shape[2], shape[3]);

            // Conversion:  shape -> CUDA extent
            // 3D:          1DHW  -> DHW
            // 2D:          11HW  -> 0HW
            // 1D:          111W  -> 00W
            // 2D layered:  B1HW  -> DHW
            // 1D layered:  B11W  -> D0W
            check(all(shape > 0) and shape[is_layered] == 1,
                  "The input shape cannot be converted to a CUDA array extent. "
                  "Dimensions with a size of 0 are not allowed, and the {} should be 1. Shape: {}",
                  is_layered ? "depth dimension (for layered arrays)" : "batch dimension", shape);

            auto shape_3d = shape.filter(static_cast<i64>(not is_layered), 2, 3).as_safe<size_t>();

            // Set empty dimensions to 0. If layered, leave extent.depth to the batch value.
            if (not is_layered)
                shape_3d[0] -= shape_3d[0] == 1;
            shape_3d[1] -= shape_3d[1] == 1;
            return {shape_3d[2], shape_3d[1], shape_3d[0]};
        }

        static Shape4<i64> extent2shape(cudaExtent extent, bool is_layered) noexcept {
            auto u_extent = Shape{extent.depth, extent.height, extent.width};
            u_extent += Shape3<size_t>::from_vec(u_extent == 0); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1d extents are interpreted as row vectors.
            auto shape = Shape4<i64>::from_values(1, 1, u_extent[1], u_extent[2]);
            shape[not is_layered] = static_cast<i64>(u_extent[0]);
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

    public: // static texture utilities
        /// Sets the underlying texture filter and addressing mode according to "interp_mode" and "border_mode".
        static auto convert_to_description(
            Interp interp_mode,
            Border border_mode
        ) -> Tuple<cudaTextureFilterMode, cudaTextureAddressMode, cudaTextureReadMode, bool> {
            // The accurate modes use nearest-lookups, while the fast methods use linear lookups.
            cudaTextureFilterMode filter_mode =
                interp_mode.is_fast() and interp_mode != Interp::NEAREST_FAST ?
                cudaFilterModeLinear : cudaFilterModePoint;

            cudaTextureAddressMode address_mode;
            bool normalized_coordinates{false};
            switch (border_mode) {
                case Border::PERIODIC: {
                    address_mode = cudaAddressModeWrap;
                    normalized_coordinates = true;
                    break;
                }
                case Border::MIRROR: {
                    address_mode = cudaAddressModeMirror;
                    normalized_coordinates = true;
                    break;
                }
                case Border::CLAMP: {
                    address_mode = cudaAddressModeClamp;
                    break;
                }
                case Border::VALUE: // not natively supported, fallback to ZERO
                case Border::REFLECT: // not natively supported, fallback to ZERO
                case Border::NOTHING: // not natively supported, fallback to ZERO
                case Border::ZERO: {
                    address_mode = cudaAddressModeBorder;
                    break;
                }
            }
            return make_tuple(filter_mode, address_mode, cudaReadModeElementType, normalized_coordinates);
        }

        /// Creates a 1d, 2d or 3d texture from a CUDA array.
        /// \param array                        CUDA array. Its lifetime should exceed the lifetime of this new object.
        /// \param filter_mode                  Filter mode, either cudaFilterModePoint or cudaFilterModeLinear.
        /// \param address_mode                 Address mode, either cudaAddressModeWrap, cudaAddressModeClamp,
        ///                                     cudaAddressModeMirror or cudaAddressModeBorder.
        /// \param normalized_reads_to_float    Whether 8-, 16-integer data should be converted to float when fetching.
        ///                                     Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        /// \param normalized_coordinates       Whether the coordinates are normalized when fetching.
        /// \note cudaAddressModeMirror and cudaAddressModeWrap are only available with normalized coordinates.
        ///       If normalized_coordinates is false, border_mode is switched (internally by CUDA) to cudaAddressModeClamp.
        static cudaTextureObject_t create_texture(
            const cudaArray* array,
            cudaTextureFilterMode filter_mode,
            cudaTextureAddressMode address_mode,
            cudaTextureReadMode normalized_reads_to_float,
            bool normalized_coordinates
        ) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = const_cast<cudaArray*>(array); // one example where we need const_cast...
            // TODO cudaArrayGetInfo can be used to extract the array type and make
            //      sure it matches T, but is it really useful? Maybe just an assert?

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = address_mode;
            tex_desc.addressMode[1] = address_mode; // ignored if 1d array.
            tex_desc.addressMode[2] = address_mode; // ignored if 1d or 2d array.
            tex_desc.filterMode = filter_mode;
            tex_desc.readMode = normalized_reads_to_float;
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture{};
            if (cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr))
                panic("Creating the texture object from a CUDA array failed");
            return texture;
        }

        /// Returns a texture object's texture descriptor.
        static cudaTextureDesc texture_description(cudaTextureObject_t texture) {
            cudaTextureDesc tex_desc{};
            check(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        /// Returns a texture object's texture descriptor.
        static cudaResourceDesc texture_resource(cudaTextureObject_t texture) {
            cudaResourceDesc tex_desc{};
            check(cudaGetTextureObjectResourceDesc(&tex_desc, texture));
            return tex_desc;
        }

        static cudaArray* texture_array(cudaTextureObject_t texture) {
            const auto array_resource = texture_resource(texture);
            check(array_resource.resType == cudaResourceTypeArray, "The texture is not bound to a CUDA array");
            return array_resource.res.array.array;
        }

        /// Whether texture is using normalized coordinates.
        static bool has_normalized_coordinates(cudaTextureObject_t texture) {
            return texture_description(texture).normalizedCoords;
        }
    };
}

#endif
