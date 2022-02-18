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
    /// \tparam T    Type of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type.
    /// \details PtrDevice can either use the device-wide allocations (e.g. cudaMalloc) or the stream-ordered
    ///          allocations (e.g. cudaMallocAsync). Stream-ordered operations are highly recommended.
    template<typename T>
    class PtrDevice {
    public: // static functions
        /// Allocates device memory using cudaMalloc.
        /// \param elements     Number of elements to allocate on the current device.
        /// \return Pointer pointing to device memory with an alignment of at least 256 bytes.
        ///         Use dealloc() or ::cudaFree() to free it.
        static NOA_HOST T* alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, elements * sizeof(T)));
            return static_cast<T*>(tmp);
        }

        /// Deallocates device memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to device memory, or nullptr.
        static NOA_HOST void dealloc(T* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }

        /// Allocates device memory asynchronously using cudaMallocAsync.
        /// \param elements         Number of elements to allocate on the current device.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        /// \return Pointer pointing to device memory with an alignment of at least 256 bytes.
        ///         Use the async dealloc() or ::cudaFreeAsync() to free it.
        static NOA_HOST T* alloc(size_t elements, Stream& stream) {
            #if CUDART_VERSION >= 11020
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocAsync(&tmp, elements * sizeof(T), stream.id()));
            return static_cast<T*>(tmp);
            #else
            return alloc(elements);
            #endif
        }

        /// Deallocates asynchronously device memory allocated by the async alloc() function.
        /// \param[out] ptr         Pointer pointing to device memory, or nullptr.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        static NOA_HOST void dealloc(T* ptr, Stream& stream) {
            #if CUDART_VERSION >= 11020
            NOA_THROW_IF(cudaFreeAsync(ptr, stream.id())); // if nullptr, it does nothing
            #else
            stream.synchronize(); // make sure all work is done before releasing to OS.
            return dealloc(ptr);
            #endif
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrDevice() = default;

        /// Allocates some \p T \p elements on the current device using cudaMalloc().
        /// \param elements     Number of elements to allocate.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        NOA_HOST explicit PtrDevice(size_t elements) : m_elements(elements) {
            m_ptr = alloc(elements);
        }

        /// Allocates some \p T \p elements using cudaMallocAsync().
        /// \param elements         Number of elements to allocate asynchronously on the stream's device.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        /// \warning The created object will be attached to \p stream, therefore requiring this stream to outlive
        ///          the created instance. When the PtrDevice destructor is called, the memory will be release to
        ///          the stream's device memory pool.
        NOA_HOST explicit PtrDevice(size_t elements, Stream& stream)
                : m_elements(elements), m_ptr(alloc(elements, stream)), m_stream(&stream) {}

        /// Creates an instance from a existing data.
        /// \param[in] data     Device pointer to hold on.
        /// \param elements     Number of \p T elements in \p data. If it is a nullptr, it should be 0.
        /// \note When the destructor is called, and unless it is released first, \p data will be passed to cudaFree().
        NOA_HOST PtrDevice(T* data, size_t elements) noexcept
                : m_elements(elements), m_ptr(data) {}

        /// Creates an instance from a existing data.
        /// \param[in] data         Device pointer to hold on.
        /// \param elements         Number of \p T elements in \p data.
        /// \param[in,out] stream   Stream attached to \p data.
        /// \warning The object will be attached to \p stream, therefore requiring this stream to outlive this object.
        ///          When the PtrDevice destructor is called, unless release() is called first, the \p data will be
        ///          release to the stream's device memory pool.
        NOA_HOST PtrDevice(T* data, size_t elements, Stream& stream) noexcept
                : m_elements(elements), m_ptr(data), m_stream(&stream) {}

        /// Move constructor. \p to_move is not meant to be used after this call.
        NOA_HOST PtrDevice(PtrDevice<T>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  m_stream(std::exchange(to_move.m_stream, nullptr)) {}

        /// Move assignment operator. \p to_move is not meant to be used after this call.
        NOA_HOST PtrDevice<T>& operator=(PtrDevice<T>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
                m_stream = std::exchange(to_move.m_stream, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrDevice(const PtrDevice<T>& to_copy) = delete;
        PtrDevice<T>& operator=(const PtrDevice<T>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr T* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr T* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* data() const noexcept { return m_ptr; }

        /// How many elements of type \p T are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(T); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        [[nodiscard]] NOA_HOST constexpr T* begin() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* begin() const noexcept { return m_ptr; }

        /// Returns a pointer pointing at the end + 1 of the managed data.
        [[nodiscard]] NOA_HOST constexpr T* end() noexcept { return m_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const T* end() const noexcept { return m_ptr + m_elements; }

        /// Returns a pointer to the stream used to allocate the managed data.
        /// If the data was created synchronously (without a stream), returns nullptr.
        [[nodiscard]] NOA_HOST constexpr Stream* stream() const noexcept { return m_stream; }

        /// Clears the underlying data, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
            m_ptr = nullptr;
            m_stream = nullptr;
        }

        /// Clears the underlying data, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = alloc(m_elements);
            m_stream = nullptr;
        }

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size_t elements, Stream& stream) {
            dealloc_();
            m_elements = elements;
            m_ptr = alloc(m_elements, stream);
            m_stream = &stream;
        }

        /// Resets the underlying data.
        /// \param[in] data     Device pointer to hold on. If it is not a nullptr, it should correspond to \p elements.
        /// \param elements     Number of \p T elements in \p data.
        NOA_HOST void reset(T* data, size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
            m_stream = nullptr;
        }

        /// Resets the underlying data.
        /// \param[in] data         Device pointer to hold on. If it is not a nullptr, it should correspond to \p elements.
        /// \param elements         Number of \p T elements in \p data.
        /// \param[in,out] stream   Stream attached to \p data.
        /// \warning The object will be attached to \p stream, therefore requiring this stream to outlive this object.
        ///          When the PtrDevice destructor is called, unless release() is called first, the \p data will be
        ///          release to the stream's device memory pool.
        NOA_HOST void reset(T* data, size_t elements, Stream& stream) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
            m_stream = &stream;
        }

        /// Releases the ownership of the managed pointer, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() and stream() returns nullptr after the call.
        [[nodiscard]] NOA_HOST T* release() noexcept {
            m_elements = 0;
            m_stream = nullptr;
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates the data.
        NOA_HOST ~PtrDevice() noexcept(false) {
            try {
                dealloc_();
            } catch (const std::exception& e) {
                if (std::uncaught_exceptions() == 0)
                    std::rethrow_exception(std::current_exception());
            }
        }

    private:
        NOA_HOST void dealloc_() const {
            if (m_stream)
                dealloc(m_ptr, *m_stream);
            else
                dealloc(m_ptr);
        }

        size_t m_elements{0};
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<T>, T*> m_ptr{nullptr};
        Stream* m_stream{nullptr};
    };
}
