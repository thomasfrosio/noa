#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry {
    template<typename Value, typename = std::enable_if_t<nt::is_any_v<Value, f32, f64, c32, c64>>>
    void cubic_bspline_prefilter(
            const Value* input, Strides4<i64> input_strides,
            Value* output, Strides4<i64> output_strides,
            Shape4<i64> shape, Stream& stream);
}
