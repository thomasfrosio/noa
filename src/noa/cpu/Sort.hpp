#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu {
    // Sorts an array, in-place.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void sort(T* array, const Strides4<i64>& strides, const Shape4<i64>& shape, bool ascending, i32 dim);
}
