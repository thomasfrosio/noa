/// \file noa/gpu/cuda/memory/PtrDevicePadded.h
/// \brief Hold memory with a padded layout on the device.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
#pragma once

#include <type_traits>
#include <string>
#include <utility>      // std::exchange
#include <cstddef>      // size_t

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// Padded layouts
//  PtrDevice is managing "linear" layouts, as opposed to PtrDevicePadded, which manages "padded" layouts. This padding
//  is on the right side of the innermost dimension. The size of the innermost dimension, including the padding, is
//  referred to as the pitch. "Padded" layouts can be useful to minimize the number of memory accesses on a given row
//  (but can increase the number of memory accesses for reading the whole array) and to reduce shared memory bank
//  conflicts. See https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
//
// Pitch
//  As a result, PtrDevicePadded has to keep track of a pitch and a shape. The pitch is stored in number of elements
//  and the entire API will always expect pitches in number of elements. This is just simpler to use but could be an
//  issue since CUDA does not guarantee the pitch (returned in bytes) to be divisible by the type alignment
//  requirement (which are actually unknown to cudaMalloc*). However, it looks like all devices will return a pitch
//  divisible by at least 16 bytes, which is the maximum size allowed by PtrDevicePadded. As a security, some tests
//  in noa_tests check this assumption holds for various sizes and PtrDevicePadded<>::alloc() will check it as well.

namespace noa::cuda::memory {
    /// Manages a device pointer. This object cannot be used on the device and is not copyable.
    /// \tparam T   Type of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type.
    template<typename T>
    class PtrDevicePadded {
    public: // static functions
        /// Allocates device memory using cudaMalloc3D.
        /// \param shape    Rightmost shape. If any dimension is 0, no allocation is performed.
        /// \return         1: Pointer pointing to device memory.
        ///                 2: Pitch of the padded layout, in number of elements.
        NOA_HOST static std::pair<T*, size_t> alloc(size4_t shape) {
            cudaExtent extent{shape[3] * sizeof(T), shape[2] * shape[1], shape[0]};
            cudaPitchedPtr pitched_ptr{};
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));

            // Check even in Release mode. This should never fail *cross-fingers*
            if (pitched_ptr.pitch % sizeof(T) != 0) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                NOA_THROW("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                          string::typeName<T>(), pitched_ptr.pitch, sizeof(T));
            }
            return {static_cast<T*>(pitched_ptr.ptr), pitched_ptr.pitch / sizeof(T)};
        }

        /// Allocates device memory using cudaMalloc3D. The outermost dimension is empty.
        NOA_HOST static std::pair<T*, size_t> alloc(size3_t shape) {
            return alloc(size4_t{1, shape[0], shape[1], shape[2]});
        }

        /// Deallocates device memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to device memory, or nullptr.
        /// \throw This function can throw if cudaFree fails (e.g. double free).
        NOA_HOST static void dealloc(T* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }

    public: // member functions
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrDevicePadded() = default;

        /// Allocates "padded" memory with a given \p shape on the current device using \c cudaMalloc3D.
        /// \param shape Rightmost shape.
        /// \note If any element of \p shape is 0, the allocation will not be performed.
        ///       For instance, to specify a 2D array, \p shape should be {1, 1, Y, X}.
        /// \note The created instance is the owner of the data.
        ///       To get a non-owning pointer, use get().
        ///       To release the ownership, use release().
        NOA_HOST explicit PtrDevicePadded(size4_t shape) : m_shape(shape) {
            std::tie(m_ptr, m_pitch) = alloc(shape);
        }

        /// Creates an instance from a existing data.
        /// \param[in] data     Device pointer to hold on.
        /// \param pitch        Innermost pitch, in elements, of \p data.
        /// \param shape        Rightmost shape, in elements, of \p data.
        NOA_HOST PtrDevicePadded(T* data, size_t pitch, size4_t shape) noexcept
                : m_shape(shape), m_pitch(pitch), m_ptr(data) {}

        /// Move constructor. \p to_move is not meant to be used after this call.
        NOA_HOST PtrDevicePadded(PtrDevicePadded<T>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_pitch(to_move.m_pitch),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \p to_move is not meant to be used after this call.
        NOA_HOST PtrDevicePadded<T>& operator=(PtrDevicePadded<T>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_pitch = to_move.m_pitch;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrDevicePadded(const PtrDevicePadded<T>& to_copy) = delete;
        PtrDevicePadded<T>& operator=(const PtrDevicePadded<T>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr T* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr T* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const T* data() const noexcept { return m_ptr; }

        /// Returns the logical rightmost shape of the managed data.
        [[nodiscard]] NOA_HOST constexpr size4_t shape() const noexcept { return m_shape; }

        /// Returns the rightmost strides that can be used to access the managed data.
        [[nodiscard]] NOA_HOST constexpr size4_t stride() const noexcept {
            return {m_pitch * m_shape[2] * m_shape[1],
                    m_pitch * m_shape[2],
                    m_pitch,
                    1};
        }

        /// Returns the rightmost pitches of the managed object.
        [[nodiscard]] NOA_HOST constexpr size3_t pitch() const noexcept {
            return {m_shape[1], m_shape[2], m_pitch};
        }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_pitch == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Clears the underlying data, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc(m_ptr);
            m_shape = 0;
            m_pitch = 0;
            m_ptr = nullptr;
        }

        /// Clears the underlying data, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size4_t shape) {
            dealloc(m_ptr);
            m_shape = shape;
            std::tie(m_ptr, m_pitch) = alloc(m_shape);
        }

        /// Resets the underlying data.
        /// \param[in] data     Device pointer to hold on.
        ///                     If it is a nullptr, \p pitch and \p shape should be 0.
        ///                     If it is not a nullptr, it should correspond to \p pitch and \p shape.
        /// \param pitch        Innermost pitch, in elements, of \p data.
        /// \param shape        Rightmost shape, in elements, of \p data.
        NOA_HOST void reset(T* data, size_t pitch, size4_t shape) {
            dealloc(m_ptr);
            m_shape = shape;
            m_pitch = pitch;
            m_ptr = data;
        }

        /// If the current instance is an owner, releases the ownership of the managed pointer, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call.
        [[nodiscard]] NOA_HOST T* release() noexcept {
            m_shape = 0;
            m_pitch = 0;
            return std::exchange(m_ptr, nullptr);
        }

        NOA_HOST ~PtrDevicePadded() noexcept(false) {
            cudaError_t err = cudaFree(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

    private:
        size4_t m_shape{}; // in elements
        size_t m_pitch{}; // in elements
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<T> && sizeof(T) <= 16, T*> m_ptr{nullptr};
    };
}
