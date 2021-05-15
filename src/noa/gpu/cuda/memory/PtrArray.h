#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

/*
 * CUDA arrays:
 *  -   They are only accessible by kernels through texture fetching or surface reading and writing.
 *  -   They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
 *      2 components). Elements are associated with a type (components have the same type), that may be signed or
 *      unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
 *  -   They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0 in the CUDA API, but PtrArray is
 *      following the Noa's shape convention (i.e. "empty" dimensions are noted as 1).
 */

namespace Noa::CUDA::Memory {
    /// A ND CUDA array of integers (excluding (u)int64_t), float or cfloat_t.
    template<typename Type>
    class PtrArray {
    private:
        size3_t m_shape{};
        std::enable_if_t<std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                         std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>,
                         cudaArray*> m_ptr{nullptr};

    public:
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrArray() = default;

        /**
         * Allocates N CUDA array with a given @a shape on the current device using @c cudaMalloc3DArray.
         * @param shape     Logical {fast, medium, slow} shape. This is attached to the underlying managed pointer
         *                  and is fixed for the entire life of the object. Use shape() to access it.
         *                  For instance, for a 2D array, @a shape should be {X, Y, 1}, with X and Y greater than 1.
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrArray(size3_t shape, uint flags = cudaArrayDefault) : m_shape(shape) { alloc_(flags); }

        /**
         * Creates an instance from a existing data.
         * @param[in] array CUDA array to hold on. If it is not a nullptr, it should correspond to @a shape.
         * @param shape     Logical {fast, medium, slow} shape of @a array
         */
        NOA_HOST PtrArray(cudaArray* array, size3_t shape) noexcept
                : m_shape(shape), m_ptr(array) {}

        /// Move constructor. @a to_move should not be used after this call.
        NOA_HOST PtrArray(PtrArray<Type>&& to_move) noexcept
                : m_shape(to_move.m_shape), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. @a to_move should not be used after this call.
        NOA_HOST PtrArray<Type>& operator=(PtrArray<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrArray(const PtrArray<Type>& to_copy) = delete;
        PtrArray<Type>& operator=(const PtrArray<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /// Returns the number of bytes of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /// Returns the number of elements of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return getElements(m_shape); }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return elements(); }

        /// Returns the shape, in elements, of the underlying CUDA array.
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /// Gets the dimensionality of the underlying CUDA array. @warning The instance should not be empty.
        [[nodiscard]] NOA_HOST constexpr uint ndim() const noexcept { return getNDim(m_shape); }
        [[nodiscard]] NOA_HOST constexpr uint rank() const noexcept { return ndim(); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Clears the underlying array, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
            m_ptr = nullptr;
        }

        /// Clears the underlying array, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(size3_t shape, uint flags = cudaArrayDefault) {
            dealloc_();
            m_shape = shape;
            alloc_(flags);
        }

        /**
         * Resets the underlying array.
         * @param[in] array CUDA array to hold on. If it is not a nullptr, it should correspond to @a shape.
         * @param shape     Logical {fast, medium, slow} shape of @a array.
         */
        NOA_HOST void reset(cudaArray* array, size3_t shape) {
            dealloc_();
            m_shape = shape;
            m_ptr = array;
        }

        /**
         * Releases the ownership of the managed array, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call and empty() returns true.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates the array.
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            cudaExtent extent{};
            switch (getNDim(m_shape)) {
                case 1: {
                    extent.width = m_shape.x;
                    break;
                }
                case 2: {
                    extent.width = m_shape.x;
                    extent.height = m_shape.y;
                    break;
                }
                case 3: {
                    extent.width = m_shape.x;
                    extent.height = m_shape.y;
                    extent.depth = m_shape.z;
                    break;
                }
            }
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, extent, flag));
        }

        NOA_HOST void dealloc_() {
            NOA_THROW_IF(cudaFreeArray(m_ptr));
        }
    };
}
