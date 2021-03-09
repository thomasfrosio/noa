#pragma once

#include <cuda_runtime.h>

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

/*
 * Linear memory vs padded memory
 * ==============================
 *
 * PtrDevice is managing "linear" memory, as opposed to PtrDevicePadded, which managed "padded" memory.
 * "Padded" memory can be useful to minimize the number of memory access on a given row (but can increase the number
 * of memory accesses for reading the whole array) and to reduce shared memory bank conflicts. This is detailed in
 * https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
 *
 * As a result, PtrDevicePadded has to keep track of a pitch and a shape (width|row, height|column and depth|page).
 */

namespace Noa::CUDA {
    /**
     * Manages a device pointer. This object cannot be used on the device and is not copyable.
     * @tparam Type     Type of the underlying pointer. Anything allowed by @c Traits::is_valid_ptr_type.
     * @throw           @c Noa::Exception, if an error occurs when the data is allocated or freed.
     */
    template<typename Type>
    class PtrDevicePadded {
    private:
        size3_t m_shape{}; // in elements
        std::enable_if_t<Noa::Traits::is_valid_ptr_type_v<Type>, Type*> m_ptr{nullptr};
        size_t m_pitch{}; // in bytes

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrDevicePadded() = default;

        /**
         * Allocates "padded" memory with a given @a shape on the current device using @c cudaMalloc3D.
         * @param shape     Either 1D, 2D or 3D. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements() * sizeof(Type)`, see bytes() and bytesPadded().
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          For instance, to specify a 2D array, @a shape should be {X, Y, 1}.
         *
         * @note    The created instance is the owner of the data.
         *          To get a non-owning pointer, use get().
         *          To release the ownership, use release().
         */
        NOA_HOST explicit PtrDevicePadded(size3_t shape) : m_shape(shape) { alloc_(); }

        /**
         * Creates an instance from a existing data.
         * @param[in] data  Device pointer to hold on.
         *                  If it is a nullptr, @a pitch and @a shape should be 0.
         *                  If it is not a nullptr, it should correspond to @a pitch and @a shape.
         * @param pitch     The pitch, in bytes, of @a data.
         * @param shape     The shape, in elements, of @a data.
         */
        NOA_HOST PtrDevicePadded(Type* data, size_t pitch, size3_t shape) noexcept
                : m_shape(shape), m_ptr(data), m_pitch(pitch) {}

        /** Move constructor. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrDevicePadded(PtrDevicePadded<Type>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  m_pitch(to_move.m_pitch) {}

        /** Move assignment operator. @a to_move is not meant to be used after this call. */
        NOA_HOST PtrDevicePadded<Type>& operator=(PtrDevicePadded<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
                m_pitch = to_move.m_pitch;
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit Memory::copy() functions.
        PtrDevicePadded(const PtrDevicePadded<Type>& to_copy) = delete;
        PtrDevicePadded<Type>& operator=(const PtrDevicePadded<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /** Returns the pitch (in bytes) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t pitch() const noexcept { return m_pitch ; }

        /// Returns the pitch (in elements) of the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t pitchElements() const noexcept { return m_pitch / sizeof(Type); }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return getElements(m_shape); }

        /** How many bytes (excluding the padding) are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /** How many bytes (including the padding) are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytesPadded() const noexcept { return m_pitch * m_shape.y * m_shape.z; }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_pitch == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
            m_ptr = nullptr;
            m_pitch = 0;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size3_t shape) {
            dealloc_();
            m_shape = shape;
            alloc_();
        }

        /**
         * Resets the underlying data.
         * @param[in] data  Device pointer to hold on.
         *                  If it is a nullptr, @a pitch and @a shape should be 0.
         *                  If it is not a nullptr, it should correspond to @a pitch and @a shape.
         * @param pitch     The pitch, in bytes, of @a data.
         * @param shape     The shape, in elements, of @a data.
         */
        NOA_HOST void reset(Type* data, size_t pitch, size3_t shape) {
            dealloc_();
            m_shape = shape;
            m_ptr = data;
            m_pitch = pitch;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_shape = size_t{0};
            m_pitch = size_t{0};
            return std::exchange(m_ptr, nullptr);
        }

        /** Deallocates owned data. */
        NOA_HOST ~PtrDevicePadded() { dealloc_(); }

    private:
        // Allocates using cudaMalloc3D. If x, y or z is 0, no allocation is performed: ptr=nullptr, pitch=0.
        NOA_HOST void alloc_() {
            cudaExtent extent{m_shape.x * sizeof(Type), m_shape.y, m_shape.z};
            cudaPitchedPtr pitched_ptr{};
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));
            m_ptr = static_cast<Type*>(pitched_ptr.ptr);
            m_pitch = pitched_ptr.pitch;
        }

        NOA_HOST void dealloc_() {
            NOA_THROW_IF(cudaFree(m_ptr)); // if nullptr, does nothing
        }
    };
}
