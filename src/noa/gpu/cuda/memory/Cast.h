#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T, typename U>
    constexpr bool is_valid_cast_v =
            ((traits::is_restricted_scalar_v<T> || std::is_same_v<T, bool>) &&
             (traits::is_restricted_scalar_v<U> || std::is_same_v<U, bool>)) ||
            (traits::is_complex_v<T> && traits::is_complex_v<U>);
}

namespace noa::cuda::memory {
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const Shared<T[]>& input,
              const Shared<U[]>& output,
              i64 elements, bool clamp, Stream& stream);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const Shared<T[]>& input, const Strides4<i64>& input_strides,
              const Shared<U[]>& output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, bool clamp, Stream& stream);
}
