#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    // Sorts an array, in-place.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void sort(const shared_t<T[]>& array, dim4_t strides, dim4_t shape, bool ascending, int dim, Stream& stream);
}
