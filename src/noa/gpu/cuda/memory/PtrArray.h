/// \file noa/gpu/cuda/memory/PtrArray.h
/// \brief Hold CUDA arrays.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include <type_traits>
#include <utility>      // std::exchange

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// CUDA arrays:
//  -   Data resides in global memory. The host can cudaMemcpy to it and the device can only access it through texture
//      reads or surface reads and writes.
//  -   They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
//      2 components). Elements are associated with a type (components have the same type), that may be signed or
//      unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
//  -   They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0 in the CUDA API, but PtrArray is
//      following noa's "shape" convention (i.e. "empty" dimensions are noted as 1).
//
// Notes:
//  - They are cache optimized for 2D/3D spatial locality.
//  - Surfaces and textures can be bound to same CUDA array.
//  - They are mostly used if the content changes rarely. Although reusing them with cudaMemcpy is possible.

namespace noa::cuda::memory {
    /// A ND CUDA array of integers (excluding (u)int64_t), float or cfloat_t.
    /// \note cfloat_t has the same channel descriptor as the CUDA built-in vector float2.
    template<typename Type>
    class PtrArray {
    public: // static functions
        NOA_HOST static cudaArray* alloc(size3_t shape, uint flag) {
            cudaExtent extent{};
            switch (ndim(shape)) {
                case 1: {
                    extent.width = shape.x;
                    break;
                }
                case 2: {
                    extent.width = shape.x;
                    extent.height = shape.y;
                    break;
                }
                case 3: {
                    extent.width = shape.x;
                    extent.height = shape.y;
                    extent.depth = shape.z;
                    break;
                }
            }
            cudaArray* ptr;
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&ptr, &desc, extent, flag));
            return ptr;
        }

        NOA_HOST static void dealloc(cudaArray* ptr) {
            NOA_THROW_IF(cudaFreeArray(ptr));
        }

    public: // member functions
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrArray() = default;

        /// Allocates a ND CUDA array with a given \p shape on the current device using \c cudaMalloc3DArray.
        /// \param shape    Logical {fast, medium, slow} shape. This is attached to the underlying managed pointer
        ///                 and is fixed for the entire life of the object. Use shape() to access it.
        ///                 For instance, for a 2D array, \p shape should be {X, Y, 1}, with X and Y greater than 1.
        ///
        /// \note    The created instance is the owner of the data.
        ///          To get a non-owning pointer, use get().
        ///          To release the ownership, use release().
        NOA_HOST explicit PtrArray(size3_t shape, uint flags = cudaArrayDefault) : m_shape(shape) {
            m_ptr = alloc(m_shape, flags);
        }

        /// Creates an instance from existing data.
        /// \param[in] array    CUDA array to hold on. If it is not a nullptr, it should correspond to \p shape.
        /// \param shape        Logical {fast, medium, slow} shape of \p array
        NOA_HOST PtrArray(cudaArray* array, size3_t shape) noexcept
                : m_shape(shape), m_ptr(array) {}

        /// Move constructor. \p to_move should not be used after this call.
        NOA_HOST PtrArray(PtrArray<Type>&& to_move) noexcept
                : m_shape(to_move.m_shape), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \p to_move should not be used after this call.
        NOA_HOST PtrArray<Type>& operator=(PtrArray<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrArray(const PtrArray<Type>& to_copy) = delete;
        PtrArray<Type>& operator=(const PtrArray<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /// Returns the number of bytes of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /// Returns the number of elements of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return noa::elements(m_shape); }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return elements(); }

        /// Returns the shape, in elements, of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Clears the underlying array, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc(m_ptr);
            m_shape = 0UL;
            m_ptr = nullptr;
        }

        /// Clears the underlying array, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(size3_t shape, uint flags = cudaArrayDefault) {
            dealloc(m_ptr);
            m_shape = shape;
            m_ptr = alloc(m_shape, flags);
        }

        /// Resets the underlying array.
        /// \param[in] array    CUDA array to hold on. If it is not a nullptr, it should correspond to \p shape.
        /// \param shape        Logical {fast, medium, slow} shape of \p array.
        NOA_HOST void reset(cudaArray* array, size3_t shape) {
            dealloc(m_ptr);
            m_shape = shape;
            m_ptr = array;
        }

        /// Releases the ownership of the managed array, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call and empty() returns true.
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates the array.
        NOA_HOST ~PtrArray() {
            cudaError_t err = cudaFreeArray(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

    private:
        size3_t m_shape{};
        std::enable_if_t<std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                         std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>,
                cudaArray*> m_ptr{nullptr};

    };
}
