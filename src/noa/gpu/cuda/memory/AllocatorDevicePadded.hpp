#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Device.hpp"

// Padded layouts / Pitch memory
//  - AllocatorDevice is allocating "linear" regions, as opposed to AllocatorDevicePadded, which allocates "padded"
//    regions. This padding is on the right side of the innermost dimension (i.e. height, in our case). The size
//    of the innermost dimension, including the padding, is called the pitch. "Padded" layouts can be useful to
//    minimize the number of memory accesses on a given row (but can increase the number of memory accesses for
//    reading the whole array) and to reduce shared memory bank conflicts.
//    It is highly recommended to use padded layouts when per-row accesses will be needed, e.g. when the array
//    is logically treated as a series of rows, because the alignment at the beginning of every row is preserved.
//    See https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
//
// Pitch
//  - As a result, AllocatorDevicePadded returns the strides (as always, this is in number of elements) of the
//    allocated region, taking into account the pitch. Note that this could be an issue since CUDA does not guarantee
//    the pitch (originally returned in bytes) to be divisible by the type alignment requirement. However, it looks
//    like all devices will return a pitch divisible by at least 16 bytes, which is the maximum size allowed by
//    AllocatorDevicePadded. As a security, AllocatorDevicePadded::allocate() will check if this assumption holds.

namespace noa::cuda::memory {
    struct AllocatorDevicePaddedDeleter {
        void operator()(void* ptr) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFree(ptr); // if nullptr, it does nothing
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // Manages a device pointer.
    template<typename T>
    class AllocatorDevicePadded {
    public:
        static_assert(!std::is_pointer_v<T> && !std::is_reference_v<T> && !std::is_const_v<T> && sizeof(T) <= 16);
        using value_type = T;
        using deleter_type = AllocatorDevicePaddedDeleter;
        using shared_type = Shared<value_type[]>;
        using unique_type = Unique<value_type[], deleter_type>;
        static constexpr size_t ALIGNMENT = 256; // this is guaranteed by cuda

    public: // static functions
        // Allocates device memory using cudaMalloc3D.
        // Returns 1: Unique pointer pointing to the device memory.
        //         2: Pitch, i.e. height stride, in number of elements.
        template<typename Integer, size_t N, std::enable_if_t<(N >= 2), bool> = true>
        static auto allocate(
                const Shape<Integer, N>& shape, // ((B)D)HW order
                Device device = Device::current()
        ) -> std::pair<unique_type, Strides<Integer, N>> {

            if (!shape.elements())
                return {};

            // Get the extents from the shape.
            const auto s_shape = shape.template as_safe<size_t>();
            const cudaExtent extent{s_shape[N - 1] * sizeof(value_type), s_shape.pop_back().elements(), 1};

            // Allocate.
            cudaPitchedPtr pitched_ptr{};
            const DeviceGuard guard(device);
            NOA_THROW_IF(cudaMalloc3D(&pitched_ptr, extent));

            // Make sure "pitch" can be converted to a number of elements.
            if (pitched_ptr.pitch % sizeof(value_type) != 0) {
                cudaFree(pitched_ptr.ptr); // ignore any error at this point
                NOA_THROW("DEV: pitch is not divisible by sizeof({}): {} % {} != 0",
                          noa::string::human<value_type>(), pitched_ptr.pitch, sizeof(value_type));
            }

            // Create the strides.
            const auto pitch = static_cast<Integer>(pitched_ptr.pitch / sizeof(value_type));
            Strides<Integer, N> strides = shape.template set<N - 1>(pitch).strides();

            return std::pair{unique_type(static_cast<value_type*>(pitched_ptr.ptr)), strides};
        }
    };
}
