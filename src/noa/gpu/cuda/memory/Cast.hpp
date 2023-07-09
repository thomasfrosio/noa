#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::memory::details {
    template<typename T, typename U>
    constexpr bool is_valid_cast_v =
            ((noa::traits::is_restricted_scalar_v<T> || std::is_same_v<T, bool>) &&
             (noa::traits::is_restricted_scalar_v<U> || std::is_same_v<U, bool>)) ||
            (noa::traits::is_complex_v<T> && noa::traits::is_complex_v<U>);
}

namespace noa::cuda::memory {
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const T* input,
              U* output,
              i64 elements, bool clamp, Stream& stream);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const T* input, const Strides4<i64>& input_strides,
              U* output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, bool clamp, Stream& stream);
}
