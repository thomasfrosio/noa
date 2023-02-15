#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    template<typename T, typename = std::enable_if_t<noa::traits::is_restricted_scalar_v<T>>>
    void arange(T* src, i64 elements, T start, T step, Stream& stream);

    template<typename T, typename = std::enable_if_t<noa::traits::is_restricted_scalar_v<T>>>
    void arange(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
                T start, T step, Stream& stream);
}
