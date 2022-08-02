/// \file noa/gpu/cuda/memory/PtrManaged.h
/// \brief Unified memory.
/// \author Thomas - ffyr2w
/// \date 19 Oct 2021

#pragma once

#include <utility> // std::exchange

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Stream.h"

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
    template<typename T>
    class PtrManaged {
    public:
        struct Deleter {
            std::weak_ptr<Stream::Core> stream{};

            void operator()(void* ptr) noexcept {
                const std::shared_ptr<Stream::Core> stream_ = stream.lock();
                [[maybe_unused]] cudaError_t err;
                if (stream_) {
                    err = cudaStreamSynchronize(stream_->handle);
                    NOA_ASSERT(err == cudaSuccess);
                }
                err = cudaFree(ptr);
                NOA_ASSERT(err == cudaSuccess);
            }
        };

    public:
        using alloc_unique_t = unique_t<T[], Deleter>;
        static constexpr size_t ALIGNMENT = 256;

    public: // static functions
        /// Allocates \p elements of managed memory using cudaMallocManaged, accessible from any stream and any device.
        static alloc_unique_t alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, elements * sizeof(T), cudaMemAttachGlobal));
            return {static_cast<T*>(tmp), Deleter{}};
        }

        /// Allocates managed memory using cudaMallocManaged.
        /// \details The allocation is initially invisible to devices, ensuring that there's no interaction with
        ///          thread's execution in the interval between the data allocation and when the data is acquired
        ///          by the stream. The program makes a guarantee that it will only access the memory on the device
        ///          from \p stream.
        /// \param elements         Number of elements to allocate.
        /// \param[in,out] stream   Stream on which to attach the memory. The returned memory should only be accessed
        ///                         by the host, and the stream's device from kernels launched with this stream.
        ///                         Note that if the NULL stream is passed, the allocation falls back to the non-
        ///                         streamed version and the memory can be accessed by any stream on any device.
        /// \return Pointer pointing at the allocated memory.
        static alloc_unique_t alloc(size_t elements, Stream& stream) {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (!stream.id())
                return alloc(elements);
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, elements * sizeof(T), cudaMemAttachHost));
            NOA_THROW_IF(cudaStreamAttachMemAsync(stream.id(), tmp));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return {static_cast<T*>(tmp), Deleter{stream.core()}};
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        constexpr PtrManaged() = default;
        constexpr /*implicit*/ PtrManaged(std::nullptr_t) {}

        /// Allocates \p T \p elements available to the host, and any stream and any device using cudaMallocManaged().
        /// \param elements     Number of elements to allocate.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        explicit PtrManaged(size_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

        /// Allocates \p T \p elements available to the host and the stream (and its device) using cudaMallocManaged().
        /// \param elements         Number of elements to allocate.
        /// \param[in,out] stream   Stream on which to attach the memory.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        /// \warning The created object will be attached to \p stream, therefore requiring this stream to outlive
        ///          the created instance. When the PtrManaged destructor is called, the memory will be released.
        explicit PtrManaged(size_t elements, Stream& stream)
                : m_ptr(alloc(elements, stream)), m_elements(elements) {}

    public: // Getters
        /// Returns the host pointer.
        [[nodiscard]] constexpr T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// Attach the lifetime of the managed object with \p alias.
        /// \details Constructs a shared_ptr which shares ownership information with the managed object,
        ///          but holds an unrelated and unmanaged pointer \p alias. If the returned shared_ptr is
        ///          the last of the group to go out of scope, it will call the stored deleter for the
        ///          managed object of this instance. However, calling get() on this shared_ptr will always
        ///          return a copy of \p alias. It is the responsibility of the programmer to make sure that
        ///          \p alias remains valid as long as the managed object exists. This functions performs no
        ///          heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename U>
        [[nodiscard]] constexpr std::shared_ptr<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        /// Returns the shape of the allocated data as a row vector.
        [[nodiscard]] constexpr size4_t shape() const noexcept { return {1, 1, 1, m_elements}; }

        /// Returns the strides of the allocated data as a C-contiguous row vector.
        [[nodiscard]] constexpr size4_t strides() const noexcept { return shape().strides(); }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a View of the allocated data as a C-contiguous row vector.
        template<typename I>
        [[nodiscard]] constexpr View<T, I> view() const noexcept { return {m_ptr.get(), shape(), strides()}; }

        /// Returns the stream handle used to allocate the managed data.
        /// If the data was created synchronously (without a stream), returns the NULL stream.
        /// If there's no managed data, returns the NULL stream.
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

        [[nodiscard]] constexpr T& front() const noexcept { return *begin(); }
        [[nodiscard]] constexpr T& back() const noexcept { return *(end() - 1); }

    public: // Accessors
        /// Returns a reference at index \p idx. There's no bound check.
        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] constexpr T& operator[](I idx) const { return m_ptr.get()[idx]; }

        /// Releases the ownership of the managed pointer, if any.
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
