#pragma once

#include <utility> // std::exchange

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

// TODO Add prefetching and advising. Since PtrManaged is currently mostly used in tests, these performance
//      improvements are not the priority.

namespace noa::cuda::memory {
    struct PtrManagedDeleter {
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

    template<typename Value>
    class PtrManaged {
    public:
        static_assert(!std::is_pointer_v<Value> && !std::is_reference_v<Value> && !std::is_const_v<Value>);
        using value_type = Value;
        using shared_type = Shared<Value[]>;
        using deleter_type = PtrManagedDeleter;
        using unique_type = Unique<Value[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by the driver

    public: // static functions
        // Allocates "elements" of managed memory using cudaMallocManaged, accessible from any stream and any device.
        static unique_type alloc(i64 elements) {
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
        // stream: Stream on which to attach the memory. The returned memory should only be accessed
        //         by the host, and the stream's device from kernels launched with this stream.
        //         Note that if the NULL stream is passed, the allocation falls back to the non-
        //         streamed version and the memory can be accessed by any stream on any device.
        static unique_type alloc(i64 elements, Stream& stream) {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (!stream.id())
                return alloc(elements);
            if (elements <= 0)
                return {};
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, static_cast<size_t>(elements) * sizeof(value_type), cudaMemAttachHost));
            NOA_THROW_IF(cudaStreamAttachMemAsync(stream.id(), tmp));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return unique_type(static_cast<value_type*>(tmp), deleter_type{stream.core()});
        }

    public:
        // Creates an empty instance. Use reset() to allocate new data.
        constexpr PtrManaged() = default;
        constexpr /*implicit*/ PtrManaged(std::nullptr_t) {}

        // Allocates elements available to the host, and any stream and any device using cudaMallocManaged().
        explicit PtrManaged(i64 elements) : m_ptr(alloc(elements)), m_elements(elements) {}

        // Allocates elements available to the host and the stream (and its device) using cudaMallocManaged().
        // The created object will be attached to stream, therefore requiring this stream to outlive
        // the created instance. When the PtrManaged destructor is called, the memory will be released.
        explicit PtrManaged(i64 elements, Stream& stream) : m_ptr(alloc(elements, stream)), m_elements(elements) {}

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
                const auto stream_ = std::get_deleter<PtrManagedDeleter>(m_ptr)->stream.lock();
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
