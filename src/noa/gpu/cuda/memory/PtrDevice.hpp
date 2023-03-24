#pragma once

#include <utility> // std::exchange

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// PtrDevice's shared ownership
//  - PtrDevice can decouple its lifetime and the lifetime of the managed device pointer.
//  - The managed device pointer can be handled like any other std::shared_ptr<T>, its memory
//    will be correctly released to the appropriate device when the reference count reaches zero.
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
    struct PtrDeviceDeleter {
        std::weak_ptr<Stream::Core> stream{};

        void operator()(void* ptr) const noexcept {
            const std::shared_ptr<Stream::Core> stream_ = stream.lock();
            [[maybe_unused]] cudaError_t err{};
            if (!stream_) {
                // The memory was allocated 1) with cudaMalloc, so cudaFree syncs the device,
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

    // Manages a device pointer.
    // PtrDevice can either use the device-wide allocations (e.g. cudaMalloc) or the stream-ordered
    // allocations (e.g. cudaMallocAsync). Stream-ordered operations are highly recommended.
    template<typename Value>
    class PtrDevice {
    public:
        static_assert(!std::is_pointer_v<Value> && !std::is_reference_v<Value> && !std::is_const_v<Value>);
        using value_type = Value;
        using shared_type = Shared<Value[]>;
        using deleter_type = PtrDeviceDeleter;
        using unique_type = Unique<Value[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by the driver

    public: // static functions
        // Allocates device memory using cudaMalloc, with an alignment of at least 256 bytes.
        static unique_type alloc(i64 elements, Device device = Device::current()) {
            if (elements <= 0)
                return {};
            const DeviceGuard guard(device);
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, static_cast<size_t>(elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp));
        }

        // Allocates device memory asynchronously using cudaMallocAsync, with an alignment of at least 256 bytes.
        static unique_type alloc(i64 elements, Stream& stream) {
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            #if CUDART_VERSION >= 11020
            NOA_THROW_IF(cudaMallocAsync(&tmp, static_cast<size_t>(elements) * sizeof(value_type), stream.id()));
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
            #else
            DeviceGuard device(stream.device());
            NOA_THROW_IF(cudaMalloc(&tmp, static_cast<size_t>(elements) * sizeof(value_type)));
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
            #endif
        }

    public:
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrDevice() = default;
        constexpr /*implicit*/ PtrDevice(std::nullptr_t) {}

        // Allocates some elements on the current device using cudaMalloc().
        explicit PtrDevice(i64 elements, Device device = Device::current())
                : m_ptr(alloc(elements, device)), m_elements(elements) {}

        // Allocates some elements using cudaMallocAsync().
        // If the stream is not empty, the deleter of the created shared object keeps a copy of the stream to
        // ensure that the stream stays allocated until the deleter is called and the memory is released to
        // the stream's device memory pool.
        explicit PtrDevice(i64 elements, Stream& stream)
                : m_ptr(alloc(elements, stream)), m_elements(elements) {}

    public: // Getters
        [[nodiscard]] constexpr value_type* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* data() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* begin() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* end() const noexcept { return m_ptr.get() + m_elements; }
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_ptr; }
        [[nodiscard]] constexpr i64 elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr i64 size() const noexcept { return m_elements; }
        [[nodiscard]] constexpr Shape4<i64> shape() const noexcept { return {1, 1, 1, m_elements}; }
        [[nodiscard]] constexpr i64 bytes() const noexcept { return m_elements * sizeof(value_type); }
        [[nodiscard]] constexpr bool is_empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !is_empty(); }

        // Attach the lifetime of the managed object with alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists. This functions performs no
        // heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename T>
        [[nodiscard]] constexpr Shared<T[]> attach(T* alias) const noexcept { return {m_ptr, alias}; }

        // Returns the stream handle used to allocate the managed data.
        // If the data was created synchronously (without a stream), returns the NULL stream.
        // If there's no managed data, returns the NULL stream.
        [[nodiscard]] cudaStream_t stream() const {
            if (m_ptr) {
                const auto stream_ = std::get_deleter<PtrDeviceDeleter>(m_ptr)->stream.lock();
                if (stream_)
                    return stream_->handle;
            }
            return nullptr;
        }

        // Releases the ownership of the managed pointer, if any.
        shared_type release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        shared_type m_ptr{};
        i64 m_elements{0};
    };
}
