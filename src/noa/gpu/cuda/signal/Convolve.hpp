#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::details {
    template<typename T, typename U>
    constexpr bool is_valid_conv_v = noa::traits::is_real_v<T> && std::is_same_v<T, U>;
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve_1d(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter, const Shape1<i64>& filter_shape, Stream& stream);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve_2d(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter, const Shape2<i64>& filter_shape, Stream& stream);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve_3d(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter, const Shape3<i64>& filter_shape, Stream& stream);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve_separable(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter_depth, i64 filter_depth_size,
            const U* filter_height, i64 filter_height_size,
            const U* filter_width, i64 filter_width_size,
            T* tmp, Strides4<i64> tmp_strides, Stream& stream);

    // TODO Move to unified API
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const U* filter, const Shape3<i64>& filter_shape, Stream& stream);
}
