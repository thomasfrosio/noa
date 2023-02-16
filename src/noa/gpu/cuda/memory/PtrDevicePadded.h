#pragma once

#include <utility>      // std::exchange

#include "noa/core/Definitions.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

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
    struct PtrDevicePaddedDeleter {
        void operator()(void* ptr) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFree(ptr); // if nullptr, it does nothing
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // Manages a device pointer.
    template<typename Value>
    class PtrDevicePadded {
    public:
        static_assert(!std::is_pointer_v<Value> && !std::is_reference_v<Value> &&
                      !std::is_const_v<Value> && sizeof(Value) <= 16);
        using value_type = Value;
        using shared_type = Shared<Value[]>;
        using deleter_type = PtrDevicePaddedDeleter;
        using unique_type = Unique<Value[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by the driver

    public: // static functions
        // Allocates device memory using cudaMalloc3D.
        // Returns 1: Pointer pointing to device memory.
        //         2: Pitch, i.e. height stride, in number of elements.
        static auto alloc(
                const Shape4<i64>& shape,
                Device device = Device::current()
        ) -> std::pair<unique_type, i64> {
            if (!shape.elements())
                return {};

            NOA_ASSERT(all(shape > 0));
            const auto s_shape = shape.as_safe<size_t>();
            cudaExtent extent{s_shape[3] * sizeof(value_type), s_shape[2] * s_shape[1], s_shape[0]};
            cudaPitchedPtr pitched_ptr{};
            const DeviceGuard guard(device);
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));

            // Check even in Release mode. This should never fail...
            if (pitched_ptr.pitch % sizeof(value_type) != 0) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                NOA_THROW("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                          string::human<value_type>(), pitched_ptr.pitch, sizeof(value_type));
            }
            return std::pair{unique_type(static_cast<value_type*>(pitched_ptr.ptr)),
                             pitched_ptr.pitch / sizeof(value_type)};
        }

        // Allocates device memory using cudaMalloc3D. The shape is DHW.
        static std::pair<unique_type, i64> alloc(const Shape3<i64>& shape, Device device = Device::current()) {
            return alloc(shape.push_front(1), device);
        }

    public: // member functions
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrDevicePadded() = default;
        constexpr /*implicit*/ PtrDevicePadded(std::nullptr_t) {}

        // Allocates "padded" memory with a given BDHW shape on the current device using cudaMalloc3D().
        explicit PtrDevicePadded(const Shape4<i64>& shape, Device device = Device::current()) : m_shape(shape) {
            std::tie(m_ptr, m_pitch) = alloc(shape, device);
        }

    public:
        // Returns the device pointer.
        [[nodiscard]] constexpr value_type* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* data() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_ptr; }
        [[nodiscard]] constexpr bool is_empty() const noexcept { return m_pitch == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !is_empty(); }
        [[nodiscard]] constexpr const Shape4<i64>& shape() const noexcept { return m_shape; }
        [[nodiscard]] constexpr Shape4<i64> physical_shape() const noexcept {
            return {m_shape[0], m_shape[1], m_shape[2], m_pitch};
        }

        // Attach the lifetime of the managed object with alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists. This functions performs no
        // heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename T>
        [[nodiscard]] constexpr Shared<T[]> attach(T* alias) const noexcept { return {m_ptr, alias}; }

        // Releases the ownership of the managed pointer, if any.
        shared_type release() noexcept {
            m_shape = 0;
            m_pitch = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        Shape4<i64> m_shape{};
        shared_type m_ptr{};
        i64 m_pitch{}; // in elements
    };
}
