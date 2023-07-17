#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// Unified memory:
//  - Managed memory is interoperable and interchangeable with device-specific allocations, such as those created
//    using the cudaMalloc() routine. All CUDA operations that are valid on device memory are also valid on managed
//    memory; the primary difference is that the host portion of a program is able to reference and access the
//    memory as well.
//
//  - If the stream used by cudaStreamAttachMemAsync is destroyed while data is associated with it, the association is
//    removed and the association reverts to the host visibility only. Since destroying a stream is an asynchronous
//    operation, the change to default association won't happen until all work in the stream has completed.
//
//  - Data movement still happens, of course. On compute capabilities >= 6.X, page faulting means that the CUDA
//    system software doesn't need to synchronize all managed memory allocations to the GPU before each kernel
//    launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing
//    the page to be automatically migrated to the GPU memory on-demand. The same thing occurs with CPU page faults.
//
//  - GPU memory over-subscription: On compute capabilities >= 6.X, applications can allocate and access more
//    managed memory than the physical size of GPU memory.

// TODO Add prefetching and advising.

namespace noa::cuda::memory {
    struct AllocatorManagedDeleter {
        std::weak_ptr<Stream::Core> stream{};

        void operator()(void* ptr) const noexcept {
            const Shared<Stream::Core> stream_ = stream.lock();
            [[maybe_unused]] cudaError_t err;
            if (stream_) {
                err = cudaStreamSynchronize(stream_->handle);
                NOA_ASSERT(err == cudaSuccess);
            }
            err = cudaFree(ptr);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    template<typename T>
    class AllocatorManaged {
    public:
        static_assert(!std::is_pointer_v<T> && !std::is_reference_v<T> && !std::is_const_v<T>);
        using value_type = T;
        using deleter_type = AllocatorManagedDeleter;
        using shared_type = Shared<value_type[]>;
        using unique_type = Unique<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by the driver

    public: // static functions
        // Allocates managed memory using cudaMallocManaged, accessible from any stream and any device.
        static unique_type allocate_global(i64 elements) {
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(
                    &tmp, static_cast<size_t>(elements) * sizeof(value_type), cudaMemAttachGlobal));
            return unique_type(static_cast<value_type*>(tmp));
        }

        // Allocates managed memory using cudaMallocManaged.
        // The allocation is initially invisible to devices, ensuring that there's no interaction with
        // thread's execution in the interval between the data allocation and when the data is acquired
        // by the stream. The program makes a guarantee that it will only access the memory on the device
        // from stream.
        // The stream on which to attach the memory should be passed. The returned memory should only be accessed
        // by the host, and the stream's device from kernels launched with this stream. Note that if the null stream
        // is passed, the allocation falls back to allocate_global() and the memory can be accessed by any stream
        // on any device.
        static unique_type allocate(i64 elements, Stream& stream) {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (!stream.id())
                return allocate_global(elements);
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, static_cast<size_t>(elements) * sizeof(value_type), cudaMemAttachHost));
            NOA_THROW_IF(cudaStreamAttachMemAsync(stream.id(), tmp));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
        }

        // Returns the stream handle used to allocate the resource.
        // If the data was created synchronously using allocate() or if it's empty, return the null stream.
        template<typename R, std::enable_if_t<noa::traits::is_any_v<R, shared_type, unique_type>, bool> = true>
        [[nodiscard]] cudaStream_t attached_stream_handle(const R& resource) const {
            if (resource) {
                const Shared<Stream::Core> stream;
                if constexpr (std::is_same_v<R, shared_type>)
                    stream = std::get_deleter<AllocatorManagedDeleter>(resource)->stream.lock();
                else
                    resource.get_deleter().stream.lock();
                if (stream)
                    return stream->handle;
            }
            return nullptr;
        }

        // Prefetches the memory region to the stream's GPU.
        static void prefetch_to_gpu(const value_type* pointer, i64 n_elements, Stream& stream) {
            NOA_THROW_IF(cudaMemPrefetchAsync(
                    pointer, static_cast<size_t>(n_elements) * sizeof(value_type),
                    stream.device().id(), stream.id()));
        }

        // Prefetches the memory region to the cpu.
        static void prefetch_to_cpu(const value_type* pointer, i64 n_elements, Stream& stream) {
            NOA_THROW_IF(cudaMemPrefetchAsync(
                    pointer, static_cast<size_t>(n_elements) * sizeof(value_type),
                    cudaCpuDeviceId, stream.id()));
        }
    };
}
