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
//  - Data movement still happens, of course. On compute capabilities >= 6.X, page faulting means that the CUDA
//    system software doesn't need to synchronize all managed memory allocations to the GPU before each kernel
//    launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing
//    the page to be automatically migrated to the GPU memory on-demand. The same thing occurs with CPU page faults.
//
//  - GPU memory over-subscription: On compute capabilities >= 6.X, applications can allocate and access more
//    managed memory than the physical size of GPU memory.

// TODO Add prefetching and advising. Since PtrManaged is currently only used in tests, these performance
//      improvements are not necessary for our use cases.

namespace noa::cuda::memory {
    template<typename T>
    class PtrManaged {
    public: // static functions
        /// Allocates managed memory, accessible from any stream and any device, using cudaMallocManaged.
        /// \param elements     Number of elements to allocate on the current device.
        /// \return             Pointer pointing to device memory.
        /// \throw This function can throw if cudaMalloc fails.
        static NOA_HOST T* alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, elements * sizeof(T), cudaMemAttachGlobal));
            return static_cast<T*>(tmp);
        }

        /// Allocates managed memory using cudaMallocManaged.
        /// \details The allocation is initially invisible to devices, ensuring that there's no interaction with
        ///          thread's execution in the interval between the data allocation and when the data is acquired
        ///          by the stream. The program makes a guarantee that it will only access the memory on the device
        ///          from \p stream. If \p stream is destroyed while data is associated with it, the association is
        ///          removed and the association reverts to the host visibility only. Since destroying a stream is an
        ///          asynchronous operation, the change to default association won't happen until all work in the
        ///          stream has completed.
        /// \param elements         Number of elements.
        /// \param[in,out] stream   Stream on which to attach the memory. The returned memory should only be accessed
        ///                         by the host, and the stream's device from kernels launched with this stream.
        ///                         Note that if the NULL stream is passed, the allocation falls back to the non-
        ///                         streamed version and the memory can be accessed by any stream on any device.
        /// \return Pointer pointing at the allocated memory.
        static NOA_HOST T* alloc(size_t elements, Stream& stream) {
            // cudaStreamAttachMemAsync: "It is illegal to attach singly to the NULL stream, because the NULL stream
            // is a virtual global stream and not a specific stream. An error will be returned in this case".
            if (!stream.id())
                return alloc(elements);
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, elements * sizeof(T), cudaMemAttachHost));
            NOA_THROW_IF(cudaStreamAttachMemAsync(stream.id(), tmp));
            stream.synchronize(); // FIXME is this necessary since cudaMemAttachHost is used?
            return static_cast<T*>(tmp);
        }

        /// Deallocates managed memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to managed memory, or nullptr.
        static NOA_HOST void dealloc(T* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrManaged() = default;

        /// Allocates \p T \p elements available to the host, and any stream and any device using cudaMallocManaged().
        /// \param elements     Number of elements to allocate.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        NOA_HOST explicit PtrManaged(size_t elements) : m_elements(elements) {
            m_ptr = alloc(elements);
        }

        /// Allocates \p T \p elements available to the host and the stream (and its device) using cudaMallocManaged().
        /// \param elements         Number of elements to allocate.
        /// \param[in,out] stream   Stream on which to attach the memory.
        /// \note To get a non-owning pointer, use get(). To release the ownership, use release().
        /// \warning The created object will be attached to \p stream, therefore requiring this stream to outlive
        ///          the created instance. When the PtrManaged destructor is called, the memory will be released.
        NOA_HOST explicit PtrManaged(size_t elements, Stream& stream)
                : m_elements(elements), m_ptr(alloc(elements, stream)), m_stream(&stream) {}

        /// Creates an instance from existing data.
        /// \param[in] data     Managed pointer to hold on.
        /// \param elements     Number of \p T elements in \p data.
        /// \note When the destructor is called, and unless it is released first, \p data will be passed to cudaFree().
        NOA_HOST PtrManaged(T* data, size_t elements) noexcept
                : m_elements(elements), m_ptr(data) {}

        /// Creates an instance from existing data.
        /// \param[in] data         Managed pointer to hold on.
        /// \param elements         Number of \p T elements in \p data. If it is a nullptr, it should be 0.
        /// \param[in,out] stream   Stream attached to \p data.
        /// \warning The object will be attached to \p stream, therefore requiring this stream to outlive this object.
        ///          When the PtrManaged destructor is called, unless release() is called first, the \p data will be
        ///          release to the stream's device memory pool.
        NOA_HOST PtrManaged(T* data, size_t elements, Stream& stream) noexcept
                : m_elements(elements), m_ptr(data), m_stream(&stream) {}

        /// Move constructor. \p to_move is not meant to be used after this call.
        NOA_HOST PtrManaged(PtrManaged<T>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  m_stream(std::exchange(to_move.m_stream, nullptr)) {}

        /// Move assignment operator. \p to_move is not meant to be used after this call.
        NOA_HOST PtrManaged<T>& operator=(PtrManaged<T>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
                m_stream = std::exchange(to_move.m_stream, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrManaged(const PtrManaged<T>& to_copy) = delete;
        PtrManaged<T>& operator=(const PtrManaged<T>& to_copy) = delete;

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

        /// Returns a reference at index \p idx. There's no bound check.
        NOA_HOST constexpr T& operator[](size_t idx) { return m_ptr[idx]; }
        NOA_HOST constexpr const T& operator[](size_t idx) const { return m_ptr[idx]; }

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
        /// \param[in] data     Managed pointer to hold on.
        /// \param elements     Number of \p T elements in \p data.
        NOA_HOST void reset(T* data, size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
            m_stream = nullptr;
        }

        /// Resets the underlying data.
        /// \param[in] data         Managed pointer to hold on.
        /// \param elements         Number of \p T elements in \p data.
        /// \param[in,out] stream   Stream attached to \p data.
        /// \warning The object will be attached to \p stream, therefore requiring this stream to outlive this object.
        ///          When the PtrManaged destructor is called, unless release() is called first, the \p data will be
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
        NOA_HOST ~PtrManaged() noexcept(false) {
            if (m_stream->id())
                m_stream->synchronize();
            cudaError_t err = cudaFree(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
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
