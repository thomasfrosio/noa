#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T>
    constexpr bool is_valid_arange_v =
            traits::is_any_v<T, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t> ||
            traits::is_float_v<T> || traits::is_complex_v<T>;
}

namespace noa::cuda::memory {
    // Returns evenly spaced values within a given interval.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    void arange(const shared_t<T[]>& src, size_t elements, T start, T step, Stream& stream);

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    void arange(const shared_t<T[]>& src, size4_t strides, size4_t shape, T start, T step, Stream& stream);
}
