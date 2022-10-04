#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    template<typename T>
    constexpr bool is_valid_sort_v = traits::is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double>;
}

namespace noa::cuda::math {
    // Sorts an array, in-place.
    template<typename T, typename = std::enable_if_t<details::is_valid_sort_v<T>>>
    void sort(const shared_t<T[]>& array, dim4_t strides, dim4_t shape, bool ascending, int32_t dim, Stream& stream);
}
