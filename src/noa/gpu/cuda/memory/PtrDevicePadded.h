/// \file noa/gpu/cuda/memory/PtrDevicePadded.h
/// \brief Hold memory with a padded layout on the device.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021
#pragma once

#include <utility>      // std::exchange

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// PtrDevicePadded's shared ownership
//  - PtrDevicePadded can decouple its lifetime and the lifetime of the managed device pointer.
//  - The managed device pointer can be handled like any other shared_ptr<T>, its memory will be correctly released
//    to the appropriate device when the reference count reaches zero.
//
// Padded layouts
//  - PtrDevice is managing "linear" layouts, as opposed to PtrDevicePadded, which manages "padded" layouts. This
//    padding is on the right side of the innermost dimension. The size of the innermost dimension, including the
//    padding, is called the pitch. "Padded" layouts can be useful to minimize the number of memory accesses on a
//    given row (but can increase the number of memory accesses for reading the whole array) and to reduce shared
//    memory bank conflicts. For allocations of 2D and 3D objects, it is highly recommended to use padded layouts.
//    Due to alignment restrictions in the hardware, this is especially true if the application will be performing
//    memory copies involving 2D or 3D objects (whether linear memory or CUDA arrays).
//    See https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
//
// Pitch
//  - As a result, PtrDevicePadded has to keep track of a pitch and a shape. The pitch is stored in number of elements
//    and the entire API will always expect pitches in number of elements. This is just simpler to use but could be an
//    issue since CUDA does not guarantee the pitch (returned in bytes) to be divisible by the type alignment
//    requirement. However, it looks like all devices will return a pitch divisible by at least 16 bytes, which is
//    the maximum size allowed by PtrDevicePadded. As a security, PtrDevicePadded<>::alloc() will check if this
//    assumption holds, even in Release mode.

namespace noa::cuda::memory {
    /// Manages a device pointer.
    template<typename T>
    class PtrDevicePadded {
    public:
        struct Deleter {
            void operator()(void* ptr) noexcept {
                cudaFree(ptr); // if nullptr, it does nothing
            }
        };

    public: // static functions
        /// Allocates device memory using cudaMalloc3D.
        /// \param shape    Rightmost shape. If any dimension is 0, no allocation is performed.
        /// \return         1: Pointer pointing to device memory.
        ///                 2: Pitch of the padded layout, in number of elements.
        static std::pair<std::shared_ptr<T[]>, size_t> alloc(size4_t shape) {
            if (!shape.elements())
                return {nullptr, 0};

            cudaExtent extent{shape[3] * sizeof(T), shape[2] * shape[1], shape[0]};
            cudaPitchedPtr pitched_ptr{};
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));

            // Check even in Release mode. This should never fail *cross-fingers*
            if (pitched_ptr.pitch % sizeof(T) != 0) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                NOA_THROW("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                          string::human<T>(), pitched_ptr.pitch, sizeof(T));
            }
            return {{static_cast<T*>(pitched_ptr.ptr), Deleter{}},
                    pitched_ptr.pitch / sizeof(T)};
        }

        /// Allocates device memory using cudaMalloc3D.
        static std::pair<std::shared_ptr<T[]>, size_t> alloc(size3_t shape) {
            return alloc(size4_t{1, shape[0], shape[1], shape[2]});
        }

    public: // member functions
        /// Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrDevicePadded() = default;
        constexpr /*implicit*/ PtrDevicePadded(std::nullptr_t) {}

        /// Allocates "padded" memory with a given rightmost \p shape on the current device using cudaMalloc3D().
        explicit PtrDevicePadded(size4_t shape) : m_shape(shape) {
            std::tie(m_ptr, m_pitch) = alloc(shape);
        }

    public:
        /// Returns the device pointer.
        [[nodiscard]] constexpr T* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr T* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr std::shared_ptr<T[]>& share() noexcept { return m_ptr; }
        [[nodiscard]] constexpr const std::shared_ptr<T[]>& share() const noexcept { return m_ptr; }

        /// Returns the logical rightmost shape of the managed data.
        [[nodiscard]] constexpr size4_t shape() const noexcept { return m_shape; }

        /// Returns the rightmost strides that can be used to access the managed data.
        [[nodiscard]] constexpr size4_t stride() const noexcept {
            return {m_pitch * m_shape[2] * m_shape[1],
                    m_pitch * m_shape[2],
                    m_pitch,
                    1};
        }

        /// Returns the rightmost pitches of the managed object.
        [[nodiscard]] constexpr size3_t pitch() const noexcept {
            return {m_shape[1], m_shape[2], m_pitch};
        }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_pitch == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Releases the ownership of the managed pointer, if any.
        [[nodiscard]] std::shared_ptr<T[]> release() noexcept {
            m_shape = 0;
            m_pitch = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(noa::traits::is_valid_ptr_type_v<T> && sizeof(T) <= 16);
        size4_t m_shape{}; // in elements
        std::shared_ptr<T[]> m_ptr{};
        size_t m_pitch{}; // in elements
    };
}
