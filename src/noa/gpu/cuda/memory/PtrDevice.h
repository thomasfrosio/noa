/// \file noa/gpu/cuda/memory/PtrDevice.h
/// \brief Hold memory with a contiguous layout on the device.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
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
    /// Manages a device pointer.
    /// \details PtrDevice can either use the device-wide allocations (e.g. cudaMalloc) or the stream-ordered
    ///          allocations (e.g. cudaMallocAsync). Stream-ordered operations are highly recommended.
    template<typename T>
    class PtrDevice {
    public:
        struct Deleter {
            Stream stream{nullptr};

            void operator()(void* ptr) noexcept {
                if (stream.empty()) {
                    cudaFree(ptr);
                    return;
                }
                #if CUDART_VERSION >= 11020
                NOA_THROW_IF(cudaFreeAsync(ptr, stream.id())); // if nullptr, it does nothing
                #else
                stream.synchronize(); // make sure all work is done before releasing to OS.
                cudaFree(ptr);
                #endif
            }
        };

    public: // static functions
        /// Allocates device memory using cudaMalloc.
        /// \param elements     Number of elements to allocate on the current device.
        /// \return Pointer pointing to device memory with an alignment of at least 256 bytes.
        static std::unique_ptr<T[], Deleter> alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, elements * sizeof(T)));
            return {static_cast<T*>(tmp), Deleter{}};
        }

        /// Allocates device memory asynchronously using cudaMallocAsync.
        /// \param elements         Number of elements to allocate on the current device.
        /// \param[in,out] stream   Stream on which the returned memory will be attached to.
        /// \return Pointer pointing to device memory with an alignment of at least 256 bytes.
        static std::unique_ptr<T[], Deleter> alloc(size_t elements, Stream& stream) {
            #if CUDART_VERSION >= 11020
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocAsync(&tmp, elements * sizeof(T), stream.id()));
            return {static_cast<T*>(tmp), Deleter{stream}};
            #else
            DeviceGuard device(stream.device());
            return {alloc(elements), Deleter{stream}};
            #endif
        }

    public:
        /// Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrDevice() = default;
        constexpr /*implicit*/ PtrDevice(std::nullptr_t) {}

        /// Allocates some \p T \p elements on the current device using cudaMalloc().
        /// \param elements     Number of elements to allocate.
        explicit PtrDevice(size_t elements) : m_ptr(alloc(elements)), m_elements(elements) {}

        /// Allocates some \p T \p elements using cudaMallocAsync().
        /// \param elements         Number of elements to allocate asynchronously on the stream's device.
        /// \param[in,out] stream   Stream on which the returned memory will be attached to.
        /// \note If the stream is not empty, the deleter of the created shared object keeps a copy of the stream to
        ///       ensure that the stream stays allocated until the deleter is called and the memory is released to
        ///       the stream's device memory pool.
        explicit PtrDevice(size_t elements, Stream& stream)
                : m_ptr(alloc(elements, stream)), m_elements(elements) {}

    public:
        /// Returns the device pointer.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr std::shared_ptr<T[]>& share() noexcept { return m_ptr; }
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        [[nodiscard]] constexpr T* begin() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* begin() const noexcept { return m_ptr.get(); }

        /// Returns a pointer pointing at the end + 1 of the managed data.
        [[nodiscard]] constexpr T* end() noexcept { return m_ptr.get() + m_elements; }
        [[nodiscard]] constexpr const T* end() const noexcept { return m_ptr.get() + m_elements; }

        /// Returns the stream used to allocate the managed data.
        /// If the data was created synchronously (without a stream), the returned stream will be empty.
        /// If there's no managed data, returns nullptr.
        [[nodiscard]] constexpr Stream* stream() const {
            if (m_ptr)
                return &std::get_deleter<Deleter>(m_ptr)->stream;
            return nullptr;
        }

        /// Releases the ownership of the managed pointer, if any.
        std::shared_ptr<T[]> release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(noa::traits::is_valid_ptr_type_v<T>);
        std::shared_ptr<T[]> m_ptr{};
        size_t m_elements{0};
    };
}
