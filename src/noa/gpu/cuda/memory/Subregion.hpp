#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::memory {
    // Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    template<typename Value, typename = std::enable_if_t<traits::is_restricted_numeric_v<Value>>>
    void extract_subregions(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            const Vec4<i64>* origins, BorderMode border_mode, Value border_value,
            Stream& stream);

    // Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    template<typename Value, typename = std::enable_if_t<traits::is_restricted_numeric_v<Value>>>
    void insert_subregions(
            const Value* subregions, Strides4<i64> subregion_strides, Shape4<i64> subregion_shape,
            Value* output, Strides4<i64> output_strides, Shape4<i64> output_shape,
            const Vec4<i64>* origins, Stream& stream);
}