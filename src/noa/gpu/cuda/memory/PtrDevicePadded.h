/// \file noa/gpu/cuda/memory/PtrDevicePadded.h
/// \brief Hold memory with a padded layout on the device.
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
#include "noa/util/string/Format.h"

// Padded layouts
// ==============
//
// PtrDevice is managing "linear" layouts, as opposed to PtrDevicePadded, which managed "padded" layouts.
// "Padded" layouts can be useful to minimize the number of memory access on a given row (but can increase the number
// of memory accesses for reading the whole array) and to reduce shared memory bank conflicts. This is detailed in
// https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
//
// Pitch
// =====
//
// As a result, PtrDevicePadded has to keep track of a pitch and a shape. The pitch is stored in number of elements
// and the entire API will always expect pitches in number of elements. This is just simpler to use but could be an
// issue since CUDA is not guaranteeing the pitch (returned in bytes) to be divisible by the type alignment
// requirement (which are actually unknown to cudaMalloc*). However, it looks like all devices will return a pitch
// divisible by at least 16 bytes, which is the maximum size allowed by PtrDevicePadded. As a security, some tests
// in noa_tests check this assumption holds for various sizes and PtrDevicePadded<>::alloc() will check it as well.

namespace noa::cuda::memory {
    /**
     * Manages a device pointer. This object cannot be used on the device and is not copyable.
     * \tparam Type     Type of the underlying pointer. Anything allowed by \c traits::is_valid_ptr_type.
     * \throw           \c noa::Exception, if an error occurs when the data is allocated or freed.
     */
    template<typename Type>
    class PtrDevicePadded {
    private:
        size3_t m_shape{}; // in elements
        size_t m_pitch{}; // in elements
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<Type> && sizeof(Type) <= 16, Type*> m_ptr{nullptr};

    public: // static functions
        /// Allocates device memory using cudaMalloc3D.
        /// \param shape    Logical {fast, medium, slow} shape. If any dimension is 0, no allocation is performed.
        /// \return         1: Pointer pointing to device memory.
        ///                 2: Pitch of the padded layout, in number of elements.
        /// \throw If the allocation fails or if the pitch returned by CUDA cannot be expressed in \a Type elements.
        static NOA_HOST std::pair<Type*, size_t> alloc(size3_t shape) {
            cudaExtent extent{shape.x * sizeof(Type), shape.y, shape.z};
            cudaPitchedPtr pitched_ptr{};
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));

            // Check even in Release mode. This should never fail *cross-fingers*
            if (pitched_ptr.pitch % sizeof(Type) != 0) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                NOA_THROW("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                          string::typeName<Type>(), pitched_ptr.pitch, sizeof(Type));
            }
            return {static_cast<Type*>(pitched_ptr.ptr), pitched_ptr.pitch / sizeof(Type)};
        }

        /// Deallocates device memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to device memory, or nullptr.
        /// \throw This function can throw if cudaFree fails (e.g. double free).
        static NOA_HOST void dealloc(Type* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }

    public: // member functions
        /// Creates an empty instance. Use reset() to allocate new data.
        PtrDevicePadded() = default;

        /// Allocates "padded" memory with a given \a shape on the current device using \c cudaMalloc3D.
        /// \param shape    Either 1D, 2D or 3D. This is attached to the underlying managed pointer
        ///                 and is fixed for the entire life of the object. Use shape() to access it.
        /// \note If any element of \a shape is 0, the allocation will not be performed.
        ///       For instance, to specify a 2D array, \a shape should be {X, Y, 1}.
        /// \note The created instance is the owner of the data.
        ///       To get a non-owning pointer, use get().
        ///        To release the ownership, use release().
        NOA_HOST explicit PtrDevicePadded(size3_t shape) : m_shape(shape) {
            std::tie(m_ptr, m_pitch) = alloc(shape);
        }

        /// Creates an instance from a existing data.
        /// \param[in] data     Device pointer to hold on.
        ///                     If it is a nullptr, \a pitch and \a shape should be 0.
        ///                     If it is not a nullptr, it should correspond to \a pitch and \a shape.
        /// \param pitch        The pitch, in elements, of \a data.
        /// \param shape        The shape, in elements, of \a data.
        NOA_HOST PtrDevicePadded(Type* data, size_t pitch, size3_t shape) noexcept
                : m_shape(shape), m_pitch(pitch), m_ptr(data) {}

        /// Move constructor. \a to_move is not meant to be used after this call.
        NOA_HOST PtrDevicePadded(PtrDevicePadded<Type>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_pitch(to_move.m_pitch),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)) {}

        /// Move assignment operator. \a to_move is not meant to be used after this call.
        NOA_HOST PtrDevicePadded<Type>& operator=(PtrDevicePadded<Type>&& to_move) noexcept {
            if (this != &to_move) {
                m_shape = to_move.m_shape;
                m_pitch = to_move.m_pitch;
                m_ptr = std::exchange(to_move.m_ptr, nullptr);
            }
            return *this;
        }

        // This object is not copyable. Use the more explicit memory::copy() functions.
        PtrDevicePadded(const PtrDevicePadded<Type>& to_copy) = delete;
        PtrDevicePadded<Type>& operator=(const PtrDevicePadded<Type>& to_copy) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_ptr; }

        /// Returns the logical shape (in elements) of the managed object.
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /// Returns the pitch (in elements) of the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t pitch() const noexcept { return m_pitch; }

        /// Returns the pitch (in bytes) of the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t pitchBytes() const noexcept { return m_pitch * sizeof(Type); }

        /// Returns the number of logical elements (excluding the padding) are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return getElements(m_shape); }
        [[nodiscard]] NOA_HOST constexpr size_t size() const noexcept { return elements(); }

        /// How many bytes (excluding the padding) are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /// How many bytes (including the padding) are pointed by the managed object.
        [[nodiscard]] NOA_HOST constexpr size_t bytesPadded() const noexcept { return pitchBytes() * getRows(m_shape); }
        [[nodiscard]] NOA_HOST constexpr size_t elementsPadded() const noexcept { return m_pitch * getRows(m_shape); }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_pitch == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Clears the underlying data, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            dealloc(m_ptr);
            m_shape = 0UL;
            m_pitch = 0;
            m_ptr = nullptr;
        }

        /// Clears the underlying data, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying data. The new data is owned.
        NOA_HOST void reset(size3_t shape) {
            dealloc(m_ptr);
            m_shape = shape;
            std::tie(m_ptr, m_pitch) = alloc(m_shape);
        }

        /// Resets the underlying data.
        /// \param[in] data     Device pointer to hold on.
        ///                     If it is a nullptr, \a pitch and \a shape should be 0.
        ///                     If it is not a nullptr, it should correspond to \a pitch and \a shape.
        /// \param pitch        The pitch, in elements, of \a data.
        /// \param shape        The shape, in elements, of \a data.
        NOA_HOST void reset(Type* data, size_t pitch, size3_t shape) {
            dealloc(m_ptr);
            m_shape = shape;
            m_pitch = pitch;
            m_ptr = data;
        }

        /// If the current instance is an owner, releases the ownership of the managed pointer, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call.
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_shape = size_t{0};
            m_pitch = size_t{0};
            return std::exchange(m_ptr, nullptr);
        }

        /// Deallocates owned data.
        NOA_HOST ~PtrDevicePadded() {
            cudaError_t err = cudaFree(m_ptr);
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }
    };
}
