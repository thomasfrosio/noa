/// \file noa/gpu/cuda/memory/PtrDevice.h
/// \brief Hold memory with a contiguous layout on the device.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

namespace noa::cuda::memory {
    /// Manages a device pointer. This object is not meant to be used from the device and is not copyable.
    /// \tparam Type    Type of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type.
    /// \throw          \c noa::Exception, if an error occurs when the data is allocated or freed.
    template<typename Type>
    class PtrDevice {
    private:
        size_t m_elements{0};
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<Type>, Type*> m_ptr{nullptr};

    public: // static functions
        /// Allocates device memory using cudaMalloc.
        /// \param elements     Number of elements to allocate on the current device.
        /// \return             Pointer pointing to device memory.
        /// \throw This function can throw if cudaMalloc fails.
        static NOA_HOST Type* alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMalloc(&tmp, elements * sizeof(Type)));
            return static_cast<Type*>(tmp);
        }

        /// Deallocates device memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to device memory, or nullptr.
        /// \throw This function can throw if cudaFree fails (e.g. double free).
        static NOA_HOST void dealloc(Type* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrDevice() = default;

        /// Allocates \a elements elements of type \a Type on the current device using \c cudaMalloc.
        /// \param elements     This is attached to the underlying managed pointer and is fixed for the entire
        ///                     life of the object. Use elements() to access it. The number of allocated bytes is
        ///                     (at least) equal to `elements * sizeof(Type)`, see bytes().
        ///
        /// \note   The created instance is the owner of the data.
        ///         To get a non-owning pointer, use get().
        ///         To release the ownership, use release().
        NOA_HOST explicit PtrDevice(size_t elements) : m_elements(elements) {
            m_ptr = alloc(elements);
        }

        /// Creates an instance from a existing data.
        /// \param elements     Number of \a Type elements in \a data.
        /// \param[in] data     Device pointer to hold on.
        ///                     If it is a nullptr, \a elements should be 0.
        ///                     If it is not a nullptr, it should correspond to \a elements.
        NOA_HOST PtrDevice(Type* data, size_t elements) noexcept
                : m_elements(elements), m_ptr(data) {}

        /// Move constructor. \a to_move is not meant to be used after this call.
        NOA_HOST PtrDevice(PtrDevice<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \a to_move is not meant to be used after this call.
        NOA_HOST PtrDevice<Type>& operator=(PtrDevice<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrDevice(const PtrDevice<Type>& to_copy) = delete;
        PtrDevice<Type>& operator=(const PtrDevice<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_ptr; }

        /// How many elements of type \a Type are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return m_elements; }

        /// How many bytes are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Returns a pointer pointing at the beginning of the managed data.
        [[nodiscard]] NOA_HOST constexpr Type* begin() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* begin() const noexcept { return m_ptr; }

        /// Returns a pointer pointing at the end + 1 of the managed data.
        [[nodiscard]] NOA_HOST constexpr Type* end() noexcept { return m_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const Type* end() const noexcept { return m_ptr + m_elements; }

        /// Clears the underlying data, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc(m_ptr);
            m_elements = 0;
            m_ptr = nullptr;
        }

        /// Clears the underlying data, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size_t elements) {
            dealloc(m_ptr);
            m_elements = elements;
            m_ptr = alloc(m_elements);
        }

        /// Resets the underlying data.
        /// \param[in] data     Device pointer to hold on. If it is not a nullptr, it should correspond to \a elements.
        /// \param elements     Number of \a Type elements in \a data.
        NOA_HOST void reset(Type* data, size_t elements) {
            dealloc(m_ptr);
            m_elements = elements;
            m_ptr = data;
        }

        /// Releases the ownership of the managed pointer, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call.
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates the data.
        NOA_HOST ~PtrDevice() {
            cudaError_t err = cudaFree(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }
    };
}
