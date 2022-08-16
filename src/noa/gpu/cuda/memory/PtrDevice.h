#pragma once

#include <utility> // std::exchange

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Stream.h"

// PtrDevice's shared ownership
//  - PtrDevice can decouple its lifetime and the lifetime of the managed device pointer.
//  - The managed device pointer can be handled like any other shared_ptr<T>, its memory will be correctly released
//    to the appropriate device when the reference count reaches zero.
//
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
    // Manages a device pointer.
    // PtrDevice can either use the device-wide allocations (e.g. cudaMalloc) or the stream-ordered
    // allocations (e.g. cudaMallocAsync). Stream-ordered operations are highly recommended.
    template<typename T>
    class PtrDevice {
    public:
        struct Deleter {
            std::weak_ptr<Stream::Core> stream{};

            void operator()(void* ptr) noexcept {
                const std::shared_ptr<Stream::Core> stream_ = stream.lock();
                [[maybe_unused]] cudaError_t err;
                if (!stream_) {
                    // The memory was allocated 1) with cudaMalloc, so cudaFree sync the device,
                    // or 2) with cudaMallocAsync but the stream was deleted, so cudaFree instead.
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

    public:
        using alloc_unique_t = unique_t<T[], Deleter>;
        static constexpr size_t ALIGNMENT = 256;

    public: // static functions
        // Allocates device memory using cudaMalloc, with an alignment of at least 256 bytes.
        static alloc_unique_t alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, elements * sizeof(T)));
            return {static_cast<T*>(tmp), Deleter{}};
        }

        // Allocates device memory asynchronously using cudaMallocAsync, with an alignment of at least 256 bytes.
        static alloc_unique_t alloc(size_t elements, Stream& stream) {
            void* tmp{nullptr}; // X** to void** is not allowed
            #if CUDART_VERSION >= 11020
            NOA_THROW_IF(cudaMallocAsync(&tmp, elements * sizeof(T), stream.id()));
            return {static_cast<T*>(tmp), Deleter{stream.core()}};
            #else
            DeviceGuard device(stream.device());
            NOA_THROW_IF(cudaMalloc(&tmp, elements * sizeof(T)));
            return {static_cast<T*>(tmp), Deleter{stream.core()}};
            #endif
        }

    public:
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrDevice() = default;
        constexpr /*implicit*/ PtrDevice(std::nullptr_t) {}

        // Allocates some elements on the current device using cudaMalloc().
        explicit PtrDevice(size_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

        // Allocates some elements using cudaMallocAsync().
        // If the stream is not empty, the deleter of the created shared object keeps a copy of the stream to
        // ensure that the stream stays allocated until the deleter is called and the memory is released to
        // the stream's device memory pool.
        explicit PtrDevice(size_t elements, Stream& stream)
                : m_ptr(alloc(elements, stream)), m_elements(elements) {}

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

        // How many elements of type T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        // Returns the shape of the allocated data as a row vector.
        [[nodiscard]] constexpr size4_t shape() const noexcept { return {1, 1, 1, m_elements}; }

        // Returns the strides of the allocated data as a C-contiguous row vector.
        [[nodiscard]] constexpr size4_t strides() const noexcept { return shape().strides(); }

        // How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        // Whether the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        // Returns a View of the allocated data as a C-contiguous row vector.
        template<typename I>
        [[nodiscard]] constexpr View<T, I> view() const noexcept { return {m_ptr.get(), shape(), strides()}; }

        // Returns the stream handle used to allocate the managed data.
        // If the data was created synchronously (without a stream), returns the NULL stream.
        // If there's no managed data, returns the NULL stream.
        [[nodiscard]] cudaStream_t stream() const {
            if (m_ptr) {
                const auto stream_ = std::get_deleter<Deleter>(m_ptr)->stream.lock();
                if (stream_)
                    return stream_->handle;
            }
            return nullptr;
        }

    public: // Iterators
        [[nodiscard]] constexpr T* begin() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* end() const noexcept { return m_ptr.get() + m_elements; }

    public: // Accessors
        // Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{};
        size_t m_elements{0};
    };
}
