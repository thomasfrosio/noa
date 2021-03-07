#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/gpu/cuda/CudaRuntime.h"
#include "noa/gpu/cuda/Exception.h"

/*
 * CUDA arrays:
 *  -   They are only accessible by kernels through texture fetching or surface reading and writing.
 *  -   They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
 *      2 components). Elements are associated with a type (components have the same type), that may be signed or
 *      unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
 *  -   They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0. Since the indexing should be
 *      known at compile time (e.g. tex2D vs tex3D), the following implementation embeds the dimension into its type.
 */

namespace Noa::CUDA {
    template<typename T, uint ndim>
    class PtrArray;

    /** A 1D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 1> {
    private:
        size_t m_elements{};
        std::enable_if_t<std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                         std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>,
                         cudaArray*> m_ptr{nullptr};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 1D CUDA array with @a elements elements on the current device using @c cudaMalloc3DArray.
         * @param elements  Number of elements. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use elements() to access it.
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrArray(size_t elements, uint flags = cudaArrayDefault)
                : m_elements(elements) { alloc_(flags); }

        /**
         * Creates an instance from a existing data.
         * @param[in] array CUDA array to hold on.
         *                  If it is a nullptr, @a elements should be 0.
         *                  If it is not a nullptr, it should correspond to @a elements.
         * @param elements  Number of @a Type elements in @a array
         */
        NOA_HOST PtrArray(cudaArray* array, size_t elements) noexcept
                : m_elements(elements), m_ptr(array) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray(PtrArray<Type, 1>&& to_move) noexcept
                : m_elements(to_move.m_elements), m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray<Type, 1>& operator=(PtrArray<Type, 1>&& to_move) noexcept {
            if (this != &to_move) {
                m_elements = to_move.m_elements;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrArray(const PtrArray<Type, 1>& to_copy) = delete;
        PtrArray<Type, 1>& operator=(const PtrArray<Type, 1>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return {m_elements, 1, 1}; }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
            m_ptr = nullptr;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size_t elements, uint flags = cudaArrayDefault) {
            dealloc_();
            m_elements = elements;
            alloc_(flags);
        }

        /**
         * Resets the underlying data.
         * @param[in] data  CUDA array to hold on.
         *                  If it is a nullptr, @a elements should be 0.
         *                  If it is not a nullptr, it should correspond to @a elements.
         * @param elements  Number of @a Type elements in @a data.
         */
        NOA_HOST void reset(cudaArray* data, size_t elements) {
            dealloc_();
            m_elements = elements;
            m_ptr = data;
        }

        /**
         * Releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call and empty() returns true.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /** Deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            if (!m_elements)
                NOA_THROW("Cannot allocate a 1D CUDA array of 0 elements");
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_elements, 0, 0), flag));
        }

        NOA_HOST void dealloc_() {
            NOA_THROW_IF(cudaFreeArray(m_ptr));
        }
    };

    /** A 2D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 2> {
    private:
        size2_t m_shape{};
        std::enable_if_t<std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                         std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>,
                         cudaArray*> m_ptr{nullptr};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 2D CUDA array with a given @a shape on the current device using @c cudaMalloc3DArray.
         * @param shape     2D shape. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it.
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          To specify a 2D CUDA array, @a shape should be {X, Y}, with X and Y > 0.
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrArray(size2_t shape, uint flags = cudaArrayDefault)
                : m_shape(shape) { alloc_(flags); }

        /**
         * Creates an instance from a existing data.
         * @param[in] array CUDA array to hold on.
         *                  If it is a nullptr, @a shape should be 0.
         *                  If it is not a nullptr, it should correspond to @a shape.
         * @param shape     Number of @a Type elements in @a array
         */
        NOA_HOST PtrArray(cudaArray* array, size2_t shape) noexcept
                : m_shape(shape), m_ptr(array) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray(PtrArray<Type, 2>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray<Type, 2>& operator=(PtrArray<Type, 2>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrArray(const PtrArray<Type, 2>& to_copy) = delete;
        PtrArray<Type, 2>& operator=(const PtrArray<Type, 2>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return {m_shape.x, m_shape.y, 1}; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return getElements(m_shape); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
            m_ptr = nullptr;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size2_t shape, uint flags = cudaArrayDefault) {
            dealloc_();
            m_shape = shape;
            alloc_(flags);
        }

        /**
         * Resets the underlying data.
         * @param[in] array     Device pointer to hold on. If it is not a nullptr, it should correspond to @a shape.
         * @param shape         Shape of the @a array.
         */
        NOA_HOST void reset(cudaArray* array, size2_t shape) {
            dealloc_();
            m_shape = shape;
            m_ptr = array;
        }

        /**
         * Releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call and empty() returns true.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /** Deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            if (!m_shape.x || !m_shape.y)
                NOA_THROW("Cannot allocate a 2D CUDA array with a dimension equal to zero. Got shape: {}", m_shape);
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_shape.x, m_shape.y, 0), flag));
        }

        NOA_HOST void dealloc_() {
            NOA_THROW_IF(cudaFreeArray(m_ptr));
        }
    };

    /** A 3D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 3> {
    private:
        size3_t m_shape{};
        std::enable_if_t<std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                         std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>,
                         cudaArray*> m_ptr{nullptr};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 3D CUDA array with a given @a shape on the current device using @c cudaMalloc3DArray.
         * @param shape     3D shape. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it.
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          To specify a 3D CUDA array, @a shape should be {X, Y, Z}, with X, Y and Z > 0.
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrArray(size3_t shape, uint flags = cudaArrayDefault)
                : m_shape(shape) { alloc_(flags); }

        /**
         * Creates an instance from a existing data.
         * @param[in] array CUDA array to hold on.
         *                  If it is a nullptr, @a shape should be 0.
         *                  If it is not a nullptr, it should correspond to @a shape.
         * @param shape     Number of @a Type elements in @a array
         */
        NOA_HOST PtrArray(cudaArray* array, size3_t shape) noexcept
                : m_shape(shape), m_ptr(array) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray(PtrArray<Type, 3>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrArray<Type, 3>& operator=(PtrArray<Type, 3>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrArray(const PtrArray<Type, 3>& to_copy) = delete;
        PtrArray<Type, 3>& operator=(const PtrArray<Type, 3>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return getElements(m_shape); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
            m_ptr = nullptr;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size3_t shape, uint flags = cudaArrayDefault) {
            dealloc_();
            m_shape = shape;
            alloc_(flags);
        }

        /**
         * Resets the underlying data.
         * @param shape         Shape of @a array.
         * @param[in] array     Device pointer to hold on. If it is not a nullptr, it should correspond to @a shape.
         */
        NOA_HOST void reset(cudaArray* array, size3_t shape) {
            dealloc_();
            m_shape = shape;
            m_ptr = array;
        }

        /**
         * Releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call and empty() returns true.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            if (!m_shape.x || !m_shape.y || !m_shape.z)
                NOA_THROW("Cannot allocate a 3D CUDA array with a dimension equal to zero. Got shape: {}", m_shape);
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_shape.x, m_shape.y, m_shape.z), flag));
        }

        NOA_HOST void dealloc_() {
            NOA_THROW_IF(cudaFreeArray(m_ptr));
        }
    };
}
