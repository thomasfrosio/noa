#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::memory {
    // Returns evenly spaced values within a given interval.
    template<typename T, typename = std::enable_if_t<noa::traits::is_restricted_numeric_v<T>>>
    T linspace(T* src, i64 elements,
               T start, T stop, bool endpoint, Stream& stream);

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T, typename = std::enable_if_t<noa::traits::is_restricted_numeric_v<T>>>
    T linspace(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
               T start, T stop, bool endpoint, Stream& stream);
}
