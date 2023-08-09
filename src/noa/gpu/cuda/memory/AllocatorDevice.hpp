#pragma once

#include <utility> // std::exchange

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// Stream-ordered allocations: (since CUDA 11.2)
//  - https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
//  - Memory allocation and deallocation cannot fail asynchronously. Memory errors that occur because of a call to
//    cudaMallocAsync or cudaFreeAsync (for example, out of memory) are reported immediately through an error code
//    returned from the call. If cudaMallocAsync completes successfully, the returned pointer is guaranteed to be
//    a valid pointer to memory that is safe to access in the appropriate stream order.
//
// Interoperability with cudaMalloc and cudaFree
//  - An application can use cudaFreeAsync to free a pointer allocated by cudaMalloc. The underlying memory is not
//    freed until the next synchronization of the stream passed to cudaFreeAsync.
//  - Similarly, an application can use cudaFree to free memory allocated using cudaMallocAsync. However, cudaFree
//    does not implicitly synchronize in this case, so the application must insert the appropriate synchronization
//    to ensure that all accesses to the to-be-freed memory are complete.

namespace noa::cuda::memory {
    // AllocatorDevice uses RAII for allocation (allocated memory lifetime is attached to the return type).
    // This is the deleter of the return type, which needs to handle asynchronous deletes.
    struct AllocatorDeviceDeleter {
        using stream_type = noa::cuda::Stream::Core;
        std::weak_ptr<stream_type> stream{};

        void operator()(void* ptr) const noexcept {
            const std::shared_ptr<stream_type> stream_ = stream.lock();
            [[maybe_unused]] cudaError_t err{};
            if (!stream_) {
                // No stream, so the memory was allocated:
                //  - with cudaMalloc, so cudaFree syncs the device
                //  - with cudaMallocAsync, but the stream was deleted, so cudaFree instead
                err = cudaFree(ptr);
            } else {
                #if CUDART_VERSION >= 11020
                err = cudaFreeAsync(ptr, stream_->handle);
                #else
                err = cudaStreamSynchronize(stream_->handle); // make sure all work is done before releasing to OS.
                NOA_ASSERT(err == cudaSuccess);
                err = cudaFree(ptr);
                #endif
            }
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // Allocates (global) device memory, using either:
    //  1) device-wide allocations via cudaMalloc.
    //  2) stream-ordered allocations via cudaMallocAsync (recommended).
    template<typename T>
    class AllocatorDevice {
    public:
        static_assert(!std::is_pointer_v<T> && !std::is_reference_v<T> && !std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorDeviceDeleter;
        using shared_type = Shared<value_type[]>;
        using unique_type = Unique<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // in bytes, this is guaranteed by cuda

    public: // static functions
        // Allocates device memory using cudaMalloc, with an alignment of at least 256 bytes.
        // This function throws if the allocation fails.
        static unique_type allocate(i64 elements, Device device = Device::current()) {
            if (elements <= 0)
                return {};
            const DeviceGuard guard(device);
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, static_cast<size_t>(elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp));
        }

        // Allocates device memory asynchronously using cudaMallocAsync, with an alignment of at least 256 bytes.
        // This function throws if the allocation fails.
        static unique_type allocate_async(i64 elements, Stream& stream) {
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            #if CUDART_VERSION >= 11020
            NOA_THROW_IF(cudaMallocAsync(&tmp, static_cast<size_t>(elements) * sizeof(value_type), stream.id()));
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
            #else
            // Async allocations didn't exist back then...
            DeviceGuard device(stream.device());
            NOA_THROW_IF(cudaMalloc(&tmp, static_cast<size_t>(elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
            #endif
        }

        // Returns the stream handle used to allocate the resource.
        // If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<typename R, std::enable_if_t<nt::is_any_v<R, shared_type, unique_type>, bool> = true>
        [[nodiscard]] static cudaStream_t attached_stream_handle(const R& resource) {
            if (resource) {
                const Shared<Stream::Core> stream;
                if constexpr (std::is_same_v<R, shared_type>)
                    stream = std::get_deleter<AllocatorDeviceDeleter>(resource)->stream.lock();
                else
                    resource.get_deleter().stream.lock();
                if (stream)
                    return stream->handle;
            }
            return nullptr;
        }
    };
}
