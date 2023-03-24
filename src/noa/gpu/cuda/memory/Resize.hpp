#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/algorithms/memory/Resize.hpp"

namespace noa::cuda::memory {
    // Resizes the input array(s) by padding and/or cropping the edges of the array.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T>>>
    void resize(const T* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
                Vec4<i64> border_left, Vec4<i64> border_right,
                T* output, Strides4<i64> output_strides,
                BorderMode border_mode, T border_value, Stream& stream);

    // Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T>>>
    void resize(const T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                BorderMode border_mode, T border_value, Stream& stream) {
        auto [border_left, border_right] = noa::algorithm::memory::borders(input_shape, output_shape);
        resize(input, input_strides, input_shape,
               border_left, border_right,
               output, output_strides,
               border_mode, border_value, stream);
    }
}
