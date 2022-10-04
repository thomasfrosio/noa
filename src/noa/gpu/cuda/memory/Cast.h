#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T, typename U>
    constexpr bool is_valid_cast_v =
            (traits::is_restricted_scalar_v<T> && traits::is_restricted_scalar_v<U>) ||
            (traits::is_complex_v<T> && traits::is_complex_v<U>);
}

namespace noa::cuda::memory {
    // Casts one array to another type.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const shared_t<T[]>& input,
              const shared_t<U[]>& output,
              dim_t elements, bool clamp, Stream& stream);

    // Casts one array to another type.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<U[]>& output, dim4_t output_strides,
              dim4_t shape, bool clamp, Stream& stream);
}
